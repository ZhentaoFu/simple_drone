import yaml
import numpy as np
import torch
from torch import nn
from torch.nn.modules import loss
from drone_env import DroneExplorationEnv
from model import Dreamer
from torch.utils.data import DataLoader, Dataset


class ExperienceDataset(Dataset):
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.obs_buf = []
        self.action_buf = []
        self.reward_buf = []

    def __len__(self):
        """返回当前存储的经验数量"""
        return len(self.obs_buf)

    def __getitem__(self, idx):
        """支持索引访问"""
        return (
            torch.FloatTensor(self.obs_buf[idx]),
            torch.FloatTensor(self.action_buf[idx]),
            torch.FloatTensor([self.reward_buf[idx]])
        )

    def add_experience(self, obs, action, reward):
        if len(self.obs_buf) >= self.capacity:
            self.obs_buf.pop(0)
            self.action_buf.pop(0)
            self.reward_buf.pop(0)
        self.obs_buf.append(obs)
        self.action_buf.append(action)
        self.reward_buf.append(reward)

def load_config():
    # train.py中load_config函数修改后
    with open("configs/dreamer.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    config = load_config()
    env = DroneExplorationEnv(config['env'])
    model = Dreamer(config)
    dataset = ExperienceDataset()

    # 定义损失函数
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for episode in range(config['training']['train_steps']):
        obs = env.reset()
        state = (torch.zeros(1, config['model']['deter_dim']),
                 torch.zeros(1, config['model']['stoch_dim']))

        # 数据收集阶段
        for _ in range(config['training']['seq_length']):
            with torch.no_grad():
                action = model.actor(torch.cat(state, dim=-1)).numpy()[0]
            next_obs, reward, done, _ = env.step(action)
            dataset.add_experience(obs, action, reward)
            obs = next_obs

            if done:
                break

        # 模型训练阶段
        # 修改后的训练循环部分
        if len(dataset) > config['training']['batch_size']:
            # 创建数据加载器
            dataloader = DataLoader(
                dataset,
                batch_size=config['training']['batch_size'],
                shuffle=True,
                drop_last=True,  # 丢弃最后不完整的批次
            )

            for batch in dataloader:
                obs_seq, action_seq, reward_seq = batch

                # 初始化隐藏状态
                deter_state = torch.zeros(config['training']['batch_size'],
                                          config['model']['deter_dim'])
                stoch_state = torch.zeros(config['training']['batch_size'],
                                          config['model']['stoch_dim'])

                # 前向传播
                states, rewards_pred, _ = model.rssm(
                    obs_seq.float(),
                    action_seq.float(),
                    (deter_state, stoch_state)
                )

                # 计算世界模型损失
                reward_loss = mse_loss(rewards_pred.squeeze(), reward_seq.float())

                # 计算价值损失
                feat = torch.cat(states, dim=-1)
                value_pred = model.value_net(feat)
                value_loss = mse_loss(value_pred.squeeze(), reward_seq.float())

                # 总损失
                total_loss = reward_loss + 0.5 * value_loss  # 调整权重系数

                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
                optimizer.step()

        # 模型保存
        if episode % config['training']['save_interval'] == 0:
            torch.save(model.state_dict(), f"dreamer_{episode}.pth")


if __name__ == "__main__":
    main()