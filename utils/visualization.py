import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(positions, target, static_obstacles, dynamic_obstacles, save_path='trajectory.png'):
    plt.figure(figsize=(8, 8))

    # 绘制无人机轨迹
    drone_x = [p[0] for p in positions]
    drone_y = [p[1] for p in positions]
    plt.plot(drone_x, drone_y, 'b-', label='Drone Path')

    # 目标位置
    plt.scatter(target[0], target[1], c='g', marker='*', s=200, label='Target')

    # 静态障碍物（直接传入坐标列表）
    for pos in static_obstacles:
        circle = plt.Circle(pos, 1.0, color='gray', alpha=0.5)
        plt.gca().add_patch(circle)

    # 动态障碍物（直接传入坐标列表） 
    for pos in dynamic_obstacles:
        circle = plt.Circle(pos, 1.0, color='orange', alpha=0.8)
        plt.gca().add_patch(circle)

    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.legend()
    plt.title('Drone Exploration Trajectory')
    plt.savefig(save_path)  # 参数化保存路径
    plt.close()