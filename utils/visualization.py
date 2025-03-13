import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(positions, target, obstacles, dynamic_obstacles):
    plt.figure(figsize=(8, 8))

    # 绘制路径
    path = np.array(positions)
    plt.plot(path[:, 0], path[:, 1], 'b.-', alpha=0.6)

    # 起点和终点
    plt.scatter(path[0, 0], path[0, 1], c='green', marker='s', label='Start')
    plt.scatter(target[0], target[1], c='red', marker='*', s=200, label='Target')

    # 静态障碍物
    for obs in obstacles:
        circle = plt.Circle(obs['pos'], 1.0, color='gray', alpha=0.5)
        plt.gca().add_patch(circle)

    # 动态障碍物
    for obs in dynamic_obstacles:
        circle = plt.Circle(obs['pos'], 1.0, color='orange', alpha=0.8)
        plt.gca().add_patch(circle)

    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.legend()
    plt.title('Drone Exploration Trajectory')
    plt.savefig('trajectory.png')
    plt.close()