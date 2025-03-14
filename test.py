import yaml
import argparse
import torch
from tqdm import tqdm
from drone_env import DroneExplorationEnv
from model import Dreamer
from utils.visualization import plot_trajectory

def parse_config():
    """解析命令行参数加载配置文件"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/dreamer.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 配置校验
    assert 'testing' in config, "Missing testing section in config"
    return config

class TestRunner:
    def __init__(self, config):
        # 初始化测试环境
        self.env = DroneExplorationEnv(config['env'])
        self.config = config
        
        # 加载训练模型
        self.model = Dreamer(config['model'])
        self.model.load_state_dict(
            torch.load(config['testing']['model_path']))
        self.model.eval()
        
        # 状态初始化
        self.initial_state = (
            torch.zeros(1, config['model']['deter_dim']),
            torch.zeros(1, config['model']['stoch_dim'])
        )

    def run_episode(self, render=False):
        """执行单次测试回合"""
        obs = self.env.reset()
        state = self.initial_state
        trajectory = {
            'positions': [obs['position']],
            'collisions': 0,
            'rewards': []
        }

        for _ in range(self.config['env']['max_steps']):
            with torch.no_grad():
                # 生成动作
                action = self.model.actor(
                    torch.cat(state, dim=-1)).numpy()[0]
            
            # 执行环境步
            next_obs, reward, done, info = self.env.step(action)
            
            # 记录轨迹数据
            trajectory['positions'].append(next_obs['position'])
            trajectory['collisions'] += int(next_obs['collision'])
            trajectory['rewards'].append(reward)
            
            # 状态更新 (网页3端到端测试理念)
            _, state = self.model.rssm.observe(
                torch.FloatTensor(next_obs['sensor']).unsqueeze(0),
                torch.FloatTensor(action).unsqueeze(0),
                state
            )
            
            # 可视化渲染
            if render and (_ % self.config['testing']['render_interval'] == 0):
                self.env.render()

            if done:
                break
        
        return trajectory

def main():
    config = parse_config()
    tester = TestRunner(config)
    
    # 执行批量测试
    results = []
    for ep in tqdm(range(config['testing']['test_episodes'])):
        render_flag = (ep % config['testing']['render_interval'] == 0)
        trajectory = tester.run_episode(render=render_flag)
        results.append(trajectory)
        
        # 生成轨迹可视化
        if config['testing']['save_trajectory']:
            plot_trajectory(
                positions=trajectory['positions'],
                target=tester.env.goal_position,
                obstacles=tester.env.static_obstacles,
                dynamic_obstacles=tester.env.dynamic_obstacles
            )
    
    # 输出测试报告
    print(f"\n=== 测试报告 ===")
    print(f"总测试回合: {len(results)}")
    print(f"平均奖励: {sum(sum(t['rewards']) for t in results)/len(results):.2f}")
    print(f"碰撞率: {sum(t['collisions']>0 for t in results)/len(results)*100:.1f}%")

if __name__ == "__main__":
    main()

def test_random_policy():
    config = {
        'grid_size': 20,
        'max_steps': 100,
        'static_obstacles': 5,
        'dynamic_obstacles': 2
    }

    env = DroneExplorationEnv(config)
    positions = []

    obs = env.reset()
    done = False
    while not done:
        action = np.random.uniform(-1, 1, 2)  # 随机策略
        obs, reward, done, _ = env.step(action)
        positions.append(env.drone_pos.copy())

    plot_trajectory(positions, env.target_pos,
                    env.static_obstacles, env.dynamic_obstacles)