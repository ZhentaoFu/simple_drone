import numpy as np
from drone_env import DroneExplorationEnv
from utils.visualization import plot_trajectory


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


if __name__ == "__main__":
    test_random_policy()


"""import torch
def evaluate_policy(env, world_model, actor_critic, num_episodes=100):
    success_count = 0
    for _ in range(num_episodes):
        obs = env.reset()
        state = world_model.init_state()

        for _ in range(env.max_steps):
            with torch.no_grad():
                action = actor_critic.act(state)
                state, _ = world_model(state, action, obs)

            obs, _, done, info = env.step(action)
            if done:
                if info['success']:
                    success_count += 1
                break

    return success_count / num_episodes"""