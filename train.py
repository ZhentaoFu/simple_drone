import yaml
import numpy as np
import torch
from torch import nn
from torch.nn.modules import loss
from drone_env import DroneExplorationEnv
from model import Dreamer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class ExperienceDataset(Dataset):
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.obs_buf = []
        self.action_buf = []
        self.reward_buf = []

    def __len__(self):
        return len(self.obs_buf)

    def __getitem__(self, idx):
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
    with open("configs/dreamer.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = DroneExplorationEnv(config['env'])
    model = Dreamer(config).to(device)
    dataset = ExperienceDataset()
    
    # 初始化混合精度训练组件
    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    mse_loss = nn.MSELoss()

    pbar_epoch = tqdm(range(config['training']['train_steps']), 
                     desc="Total Training", position=0, dynamic_ncols=True)

    for episode in pbar_epoch:
        obs = env.reset()
        state = (
            torch.zeros(1, config['model']['deter_dim'], device=device),
            torch.zeros(1, config['model']['stoch_dim'], device=device)
        )

        # 数据收集阶段
        collect_bar = tqdm(range(config['training']['seq_length']), 
                          desc=f"Episode {episode} Collecting", position=1, leave=False)
        for _ in collect_bar:
            with torch.no_grad():
                action = model.actor(torch.cat(state, dim=-1)).cpu().numpy()[0]
            next_obs, reward, done, _ = env.step(action)
            dataset.add_experience(obs, action, reward)
            obs = next_obs
            collect_bar.update(1)
            if done:
                break

        # 模型训练阶段
        if len(dataset) > config['training']['batch_size']:
            dataloader = DataLoader(dataset, 
                                  batch_size=config['training']['batch_size'],
                                  shuffle=True, 
                                  num_workers=4,
                                  pin_memory=True)

            train_bar = tqdm(dataloader, desc="Training Batches", position=2, leave=False)
            
            for batch in train_bar:
                # 数据转换为适合混合精度的格式
                obs_seq = batch[0].to(device, non_blocking=True)
                action_seq = batch[1].to(device, non_blocking=True)
                reward_seq = batch[2].to(device, non_blocking=True).squeeze(-1)  # 修复维度问题

                # 初始化隐藏状态
                deter_state = torch.zeros(config['training']['batch_size'],
                                        config['model']['deter_dim'], 
                                        device=device)
                stoch_state = torch.zeros(config['training']['batch_size'],
                                        config['model']['stoch_dim'],
                                        device=device)

                # 混合精度训练
                with torch.autocast(device_type=device.type, enabled=True):
                    states, rewards_pred, _ = model.rssm(
                        obs_seq.float(),
                        action_seq.float(),
                        (deter_state, stoch_state)
                    )
                    
                    # 统一维度
                    feat = torch.cat(states, dim=-1)
                    value_pred = model.value_net(feat)
                    
                    # 计算损失（自动处理类型转换）
                    reward_loss = mse_loss(rewards_pred.squeeze(), reward_seq.float())
                    value_loss = mse_loss(value_pred.squeeze(), reward_seq.float())
                    total_loss = reward_loss + 0.5 * value_loss

                # 梯度处理
                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                train_bar.set_postfix(loss=total_loss.item())

        # 模型保存
        if episode % config['training']['save_interval'] == 0:
            torch.save(model.state_dict(), f"dreamer_{episode}.pth")

if __name__ == "__main__":
    main()