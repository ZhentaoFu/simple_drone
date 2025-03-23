import yaml
import torch
import numpy as np
from tqdm import tqdm
from drone_env import DroneExplorationEnv
from model import Dreamer
from utils.visualization import plot_trajectory



def load_config():
    with open("configs/dreamer.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

class TestRunner:
    def __init__(self, config):
        # 设备初始化
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 环境初始化
        self.env = DroneExplorationEnv(config['env'])
        self.config = config
        self.model = Dreamer(config).to(self.device)
        
        try:
            state_dict = torch.load(
                config['testing']['model_path'],
                map_location=self.device
            )
            self.model.load_state_dict(state_dict)
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")
        
        self.model.eval()
        
        # 初始状态模板（避免跨episode污染）
        self.initial_state = (
            torch.zeros(1, config['model']['deter_dim'], device=self.device),
            torch.zeros(1, config['model']['stoch_dim'], device=self.device)
        )

    def _prepare_inputs(self, obs, action):
        """根据数组结构提取传感器数据（局部障碍物网格）"""
        # 提取局部障碍物网格（后25个元素）
        sensor_data = obs  # 转为5x5网格
        
        sensor_tensor = torch.as_tensor(
            sensor_data, 
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)  # 添加批次维度
        
        action_tensor = torch.as_tensor(
            action, 
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        return sensor_tensor, action_tensor

    def run_episode(self, render=False):
        """优化后的测试回合执行"""
        obs = self.env.reset()
        
        state = (
            self.initial_state[0].clone(),
            self.initial_state[1].clone()
        )
        
        drone_pos = obs[:2]
        trajectory = {
            'drone_positions': [drone_pos.copy()],  # 记录无人机轨迹
            'target_positions': [obs[2:4].copy()],  # 目标位置
            'obstacle_masks': [],
            'collisions': 0,
            'rewards': []
        }

        for step in range(self.config['env']['max_steps']):
            with torch.no_grad():
                # 混合精度推理
                with torch.autocast(device_type=self.device.type, enabled=True):
                    # 生成动作
                    action = self.model.actor(
                        torch.cat(state, dim=-1)
                    ).cpu().numpy()[0]
            
            # 执行环境步
            next_obs, reward, done, info = self.env.step(action)
            
             # 轨迹更新（增加障碍物状态记录）
            trajectory['drone_positions'].append(np.copy(next_obs[:2]))
            trajectory['target_positions'].append(np.copy(next_obs[2:4]))
            trajectory['obstacle_masks'].append(np.copy(next_obs[4:].reshape(5,5)))
            trajectory['collisions'] += self._check_collision(next_obs)
            trajectory['rewards'].append(float(reward)) 
            
            # 状态更新（使用RSSM前向传播）
            sensor_tensor, action_tensor = self._prepare_inputs(next_obs, action)
            with torch.autocast(device_type=self.device.type, enabled=True):
                (deter, stoch), _, _ = self.model.rssm(
                    sensor_tensor,
                    action_tensor,
                    state
                )
                state = (deter.detach(), stoch.detach())  # 断开计算图
            
            # 条件渲染优化
            if render and (step % self.config['testing'].get('render_interval', 5) == 0):
                config = load_config()
                self.env = DroneExplorationEnv(config['env'], render_mode='human')

            if done:
                break

        self.env.close()
        
        return trajectory
    
    def _check_collision(self, obs):
        """根据局部网格判断是否碰撞"""
        grid = obs[4:].reshape(5, 5)
        # 中心点（无人机当前位置）是否有障碍物
        return grid[2, 2] > 0.5

def main():
    config = load_config()
    tester = TestRunner(config)
    
    # 执行批量测试
    results = []
    try:
        for ep in tqdm(range(config['testing']['test_episodes']), desc="测试进度"):
            render_flag = (ep % config['testing'].get('render_interval', 10) == 0)
            trajectory = tester.run_episode(render=render_flag)
            results.append(trajectory)
            
            # 轨迹可视化优化
            if config['testing'].get('save_trajectory', False):
                current_obstacles = tester.env.get_obstacles()
                plot_trajectory(
                    positions=trajectory['drone_positions'],
                    target=tester.env.target_pos,
                    static_obstacles=current_obstacles['static'],
                    dynamic_obstacles=current_obstacles['dynamic'],
                    save_path=f"trajectory_{ep}.png"
                )
    finally:
        # 资源释放
        tester.env.close()
        torch.cuda.empty_cache()
    
    # 增强报告生成
    total_rewards = [sum(t['rewards']) for t in results]
    collision_rate = sum(t['collisions'] > 0 for t in results) / len(results)
    
    print(f"\n=== 综合测试报告 ===")
    print(f"总测试回合数: {len(results)}")
    print(f"平均累计奖励: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"碰撞发生率: {collision_rate*100:.1f}%")
    print(f"最大单回合奖励: {max(total_rewards):.2f}")
    print(f"最小单回合奖励: {min(total_rewards):.2f}")

if __name__ == "__main__":
    main()

def test_random_policy():
    """优化随机策略测试"""
    config = {
        'grid_size': 20,
        'max_steps': 100,
        'static_obstacles': 5,
        'dynamic_obstacles': 2
    }

    env = DroneExplorationEnv(config)
    trajectory = {
        'positions': [],
        'collisions': 0,
        'rewards': []
    }

    try:
        obs = env.reset()
        done = False
        while not done:
            action = np.random.uniform(-1, 1, 2)
            obs, reward, done, _ = env.step(action)
            trajectory['positions'].append(obs['position'].copy())
            trajectory['collisions'] += int(obs['collision'])
            trajectory['rewards'].append(reward)
        
        plot_trajectory(
            positions=trajectory['positions'],
            target=env.goal_position,
            obstacles=env.static_obstacles,
            dynamic_obstacles=env.dynamic_obstacles,
            save_path="random_trajectory.png"
        )
    finally:
        env.close()