import numpy as np
import gym
from gym import spaces
import pygame


class DroneExplorationEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], "render_fps": 30}

    def __init__(self, config, render_mode=None):
        super().__init__()
        # 环境基础配置
        self.grid_size = config['grid_size']
        self.max_steps = config['max_steps']
        self.obs_radius = 2.0  # 观测半径
        self.drone_pos = np.zeros(2)  # 初始化为零向量
        self.target_pos = np.zeros(2)
        self.steps = 0
        self.render_mode = render_mode

        # 动作空间：连续二维速度向量
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # 观测空间：无人机位置(2)+目标位置(2)+局部障碍物信息(5x5)
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size, shape=(29,), dtype=np.float32)

        # 初始化环境状态
        self.static_obstacles = self._generate_obstacles(config['static_obstacles'])
        self.dynamic_obstacles = self._generate_obstacles(config['dynamic_obstacles'], dynamic=True)

        # 渲染系统初始化
        self.screen_size = (800, 600)
        self.grid_scale = self.screen_size[0] / self.grid_size  # 像素/单位
        self.window = None
        self.clock = None
        self.trajectory = []

        # 颜色配置
        self.colors = {
            'drone': (255, 0, 0),
            'target': (0, 200, 0),
            'static_obs': (100, 100, 100),
            'dynamic_obs': (255, 165, 0),
            'trajectory': (30, 144, 255),
            'grid': (200, 200, 200)
        }

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
        self.trajectory = [self.drone_pos.copy()]
        return self._get_obs()

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
    
    def get_obstacles(self):
        """返回当前时刻障碍物状态"""
        return {
            'static': [o['pos'].copy() for o in self.static_obstacles],
            'dynamic': [o['pos'].copy() for o in self.dynamic_obstacles]
        }

    def render(self):
        if self.render_mode is None:
            return

        if self.window is None and self.render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("Drone Exploration")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.screen_size)
        canvas.fill((255, 255, 255))

        # 绘制网格系统
        for x in range(0, self.grid_size + 1):
            pygame.draw.line(canvas, self.colors['grid'],
                             (x * self.grid_scale, 0),
                             (x * self.grid_scale, self.screen_size[1]))
        for y in range(0, self.grid_size + 1):
            pygame.draw.line(canvas, self.colors['grid'],
                             (0, y * self.grid_scale),
                             (self.screen_size[0], y * self.grid_scale))

        # 绘制静态障碍物
        for obs in self.static_obstacles:
            pos = obs['pos'] * self.grid_scale
            pygame.draw.circle(canvas, self.colors['static_obs'],
                               pos.astype(int), int(1.0 * self.grid_scale))

        # 绘制动态障碍物
        for obs in self.dynamic_obstacles:
            pos = obs['pos'] * self.grid_scale
            pygame.draw.circle(canvas, self.colors['dynamic_obs'],
                               pos.astype(int), int(1.0 * self.grid_scale))

        # 绘制目标位置
        target_pixel = self.target_pos * self.grid_scale
        pygame.draw.circle(canvas, self.colors['target'],
                           target_pixel.astype(int), int(0.5 * self.grid_scale))

        # 绘制无人机轨迹
        if len(self.trajectory) > 1:
            points = [(p * self.grid_scale).astype(int) for p in self.trajectory]
            pygame.draw.lines(canvas, self.colors['trajectory'], False, points, 3)

        # 绘制无人机本体
        drone_center = self.drone_pos * self.grid_scale
        pygame.draw.circle(canvas, self.colors['drone'],
                           drone_center.astype(int), int(0.8 * self.grid_scale))

        # 状态信息叠加
        font = pygame.font.Font(None, 24)
        text = font.render(
            f"Steps: {self.steps}/{self.max_steps} | " +
            f"Position: ({self.drone_pos[0]:.1f}, {self.drone_pos[1]:.1f})",
            True, (0, 0, 0))
        canvas.blit(text, (10, 10))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()




