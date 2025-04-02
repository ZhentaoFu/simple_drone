import yaml
import torch
import numpy as np
from tqdm import tqdm
from drone_env import DroneExplorationEnv
from model import Dreamer
from utils.visualization import plot_trajectory
from gym import spaces


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


class RandomTestRunner:  # 新建随机策略测试运行器
    def __init__(self, config):
        # 设备初始化（仅使用CPU）
        self.device = torch.device("cpu")  # 随机策略无需GPU加速

        # 环境初始化
        self.env = DroneExplorationEnv(config['env'])
        self.config = config

        # 动作空间参数
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_low = self.action_space.low  # 直接获取下限[-1., -1.]
        self.action_high = self.action_space.high  # 直接获取上限[1., 1.]

    def run_episode(self, render=False):
        """随机策略的回合执行"""
        obs = self.env.reset()

        drone_pos = obs[:2]
        trajectory = {
            'drone_positions': [drone_pos.copy()],
            'target_positions': [obs[2:4].copy()],
            'obstacle_masks': [],
            'collisions': 0,
            'rewards': []
        }

        for step in range(self.config['env']['max_steps']):
            # 生成随机动作
            action = self.action_space.sample()

            # 执行环境步
            next_obs, reward, done, info = self.env.step(action)

            # 轨迹更新（与原始代码保持相同记录方式）
            trajectory['drone_positions'].append(np.copy(next_obs[:2]))
            trajectory['target_positions'].append(np.copy(next_obs[2:4]))
            trajectory['obstacle_masks'].append(np.copy(next_obs[4:].reshape(5, 5)))
            trajectory['collisions'] += self._check_collision(next_obs)
            trajectory['rewards'].append(float(reward))

            # 条件渲染（逻辑与原始代码一致）
            if render and (step % self.config['testing'].get('render_interval', 5) == 0):
                self.env.render()

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

    testers = {
        "Dreamer": TestRunner(config),
        "Random": RandomTestRunner(config)  # 新增随机策略测试器
    }

    results = {}
    try:
        for policy_name, tester in testers.items():
            policy_results = []
            for ep in tqdm(range(config['testing']['test_episodes']),
                           desc=f"{policy_name}策略测试"):
                render_flag = (ep % config['testing'].get('render_interval', 10) == 0)
                trajectory = tester.run_episode(render=render_flag)
                policy_results.append(trajectory)

                # 轨迹可视化（与原始代码逻辑相同）
                if config['testing'].get('save_trajectory', False):
                    current_obstacles = tester.env.get_obstacles()
                    plot_trajectory(
                        positions=trajectory['drone_positions'],
                        target=tester.env.target_pos,
                        static_obstacles=current_obstacles['static'],
                        dynamic_obstacles=current_obstacles['dynamic'],
                        save_path=f"{policy_name}_trajectory_{ep}.png"  # 添加策略名前缀
                    )
            results[policy_name] = policy_results
    finally:
        # 统一释放资源
        for tester in testers.values():
            tester.env.close()
        torch.cuda.empty_cache()

    # 对比报告生成
    print("\n=== 策略对比报告 ===")
    for policy_name, policy_results in results.items():
        total_rewards = [sum(t['rewards']) for t in policy_results]
        collision_rate = sum(t['collisions'] > 0 for t in policy_results) / len(policy_results)

        print(f"\n【{policy_name}策略】")
        print(f"  平均累计奖励: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        print(f"  碰撞发生率: {collision_rate * 100:.1f}%")
        print(f"  最大奖励: {max(total_rewards):.2f} | 最小奖励: {min(total_rewards):.2f}")


if __name__ == "__main__":
    main()
