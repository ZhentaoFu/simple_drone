import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class RSSM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.stoch_dim = config['model']['stoch_dim']
        self.deter_dim = config['model']['deter_dim']
        self.hidden_dim = config['model']['hidden_dim']

        # 网络定义
        self.encoder = nn.Sequential(
            nn.Linear(config['model']['obs_dim'], self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        )
        
        self.gru = nn.GRUCell(self.hidden_dim // 2 + self.stoch_dim, self.deter_dim)
        self.post_fc = nn.Linear(self.deter_dim, 2 * self.stoch_dim)
        self.prior_fc = nn.Linear(self.deter_dim, 2 * self.stoch_dim)
        
        self.reward_net = nn.Sequential(
            nn.Linear(self.deter_dim + self.stoch_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, obs, action, prev_state):
        batch_size = obs.size(0) if obs is not None else prev_state[0].size(0)
        deter_state = prev_state[0][:batch_size].detach().clone().to(self.device)
        stoch_state = prev_state[1][:batch_size].detach().clone().to(self.device)

        with torch.autocast(device_type=self.device.type, enabled=True):
            encoded = self.encoder(obs.to(self.device)) if obs is not None else None
            
            # GRU处理
            gru_input = torch.cat([encoded, stoch_state], dim=-1)
            deter_state = self.gru(gru_input, deter_state)
            
            # 后验分布
            post_mean, post_std = torch.chunk(self.post_fc(deter_state), 2, dim=-1)
            stoch_state = post_mean + torch.randn_like(post_std) * post_std
            
            # 奖励预测
            feat = torch.cat([deter_state, stoch_state], dim=-1)
            reward_pred = self.reward_net(feat)

        return (deter_state, stoch_state), reward_pred, (None, None)

class Dreamer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.config = config
        self.rssm = RSSM(config)
        self.actor = nn.Sequential(
            nn.Linear(config['model']['deter_dim'] + config['model']['stoch_dim'],
                      config['model']['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['model']['hidden_dim'], config['model']['action_dim'])
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(config['model']['deter_dim'] + config['model']['stoch_dim'],
                      config['model']['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['model']['hidden_dim'], 1)
        )

    def imagine_rollout(self, initial_state, horizon, progress=False):
        states = []
        actions = []
        rewards = []
        current_state = (
            initial_state[0].to(self.device),
            initial_state[1].to(self.device)
        )
        
        iterator = range(horizon)
        if progress:
            iterator = tqdm(iterator, desc="Imagination Rollout", leave=False)
        
        for _ in iterator:
            with torch.no_grad():
                with torch.autocast(device_type=self.device.type, enabled=True):
                    action = self.actor(torch.cat(current_state, dim=-1))
                    next_state, reward_pred, _ = self.rssm(None, action, current_state)
                
                states.append(current_state)
                actions.append(action)
                rewards.append(reward_pred)
                current_state = (
                    next_state[0].detach(),
                    next_state[1].detach()
                )
        
        return states, actions, rewards