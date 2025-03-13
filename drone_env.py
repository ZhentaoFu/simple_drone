# drone_env.py
import numpy as np
import gym
from gym import spaces


class DroneExplorationEnv(gym.Env):
    def __init__(self, config):
        self.grid_size = config['grid_size']
        self.max_steps = config['max_steps']
        self.obs_radius = 2.0  # 观测半径

        # 动作空间：连续二维速度向量
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # 观测空间：无人机位置(2)+目标位置(2)+局部障碍物信息(5x5)
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size, shape=(29,), dtype=np.float32)

        self.static_obstacles = self._generate_obstacles(config['static_obstacles'])
        self.dynamic_obstacles = self._generate_obstacles(config['dynamic_obstacles'], dynamic=True)

    def _generate_obstacles(self, num, dynamic=False):
        obstacles = []
        for _ in range(num):
            pos = np.random.uniform(0, self.grid_size, size=2)
            vel = np.random.uniform(-0.5, 0.5, size=2) if dynamic else None
            obstacles.append({'pos': pos, 'vel': vel})
        return obstacles

    def reset(self):
        self.drone_pos = np.random.uniform(0, self.grid_size, size=2)
        self.target_pos = np.random.uniform(0, self.grid_size, size=2)
        self.steps = 0
        return self._get_obs()

    # drone_env.py
    def _get_obs(self):
        # 分别生成x和y轴的坐标点
        x_coords = np.linspace(self.drone_pos[0] - self.obs_radius,
                               self.drone_pos[0] + self.obs_radius,
                               5)
        y_coords = np.linspace(self.drone_pos[1] - self.obs_radius,
                               self.drone_pos[1] + self.obs_radius,
                               5)

        # 生成网格坐标
        xx, yy = np.meshgrid(x_coords, y_coords)
        grid_points = np.stack([xx, yy], axis=-1)  # 形状(5,5,2)

        local_grid = np.zeros(25)
        index = 0
        for i in range(5):
            for j in range(5):
                pos = grid_points[i, j]
                collision = any(self._check_collision(pos, o)
                                for o in self.static_obstacles + self.dynamic_obstacles)
                local_grid[index] = 1.0 if collision else 0.0
                index += 1

        return np.concatenate([
            self.drone_pos,
            self.target_pos,
            local_grid
        ])

    def _check_collision(self, pos, obstacle):
        return np.linalg.norm(pos - obstacle['pos']) < 1.0  # 障碍物半径1.0

    def _update_dynamic_obstacles(self):
        for obs in self.dynamic_obstacles:
            # 带随机扰动的运动
            obs['pos'] += obs['vel'] + np.random.normal(0, 0.05, 2)
            # 边界反弹
            obs['pos'] = np.clip(obs['pos'], 0, self.grid_size)
            if np.any(obs['pos'] <= 0) or np.any(obs['pos'] >= self.grid_size):
                obs['vel'] *= -1

    def step(self, action):
        self._update_dynamic_obstacles()

        # 执行动作
        new_pos = self.drone_pos + np.clip(action, -1, 1) * 0.8
        new_pos = np.clip(new_pos, 0, self.grid_size)

        # 碰撞检测
        collision = any(self._check_collision(new_pos, o)
                        for o in self.static_obstacles + self.dynamic_obstacles)

        # 奖励计算
        target_dist = np.linalg.norm(new_pos - self.target_pos)
        reward = -0.1 * target_dist  # 距离奖励
        reward += 10.0 if target_dist < 1.0 else 0.0  # 到达奖励
        reward -= 5.0 if collision else 0.0  # 碰撞惩罚

        self.drone_pos = new_pos if not collision else self.drone_pos
        self.steps += 1
        done = collision or (self.steps >= self.max_steps) or (target_dist < 1.0)

        return self._get_obs(), reward, done, {}


