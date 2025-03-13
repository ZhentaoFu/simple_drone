# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class RSSM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.stoch_dim = config['model']['stoch_dim']
        self.deter_dim = config['model']['deter_dim']
        self.hidden_dim = config['model']['hidden_dim']

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(config['model']['obs_dim'], self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        )

        # GRU
        self.gru = nn.GRUCell(self.hidden_dim // 2 + self.stoch_dim, self.deter_dim)

        # Posterior
        self.post_fc = nn.Linear(self.deter_dim, 2 * self.stoch_dim)

        # Prior
        self.prior_fc = nn.Linear(self.deter_dim, 2 * self.stoch_dim)

        # Reward predictor
        self.reward_net = nn.Sequential(
            nn.Linear(self.deter_dim + self.stoch_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, obs, action, prev_state):
        # 动态获取批次维度
        batch_size = obs.size(0)

        # 调整历史状态维度
        deter_state = prev_state[0][:batch_size].clone().detach()
        stoch_state = prev_state[1][:batch_size].clone().detach()

        # 编码处理
        encoded = self.encoder(obs)

        # 维度对齐验证
        assert stoch_state.shape == (batch_size, self.stoch_dim), \
            f"State维度错误: {stoch_state.shape} vs {(batch_size, self.stoch_dim)}"

        # 拼接输入
        prior_input = torch.cat([encoded, stoch_state], dim=-1)

        # GRU处理
        deter_state = self.gru(prior_input, deter_state)

        # Prior
        prior_input = torch.cat([encoded, prev_state[1]], dim=-1)
        deter_state = self.gru(prior_input, prev_state[0])
        prior_mean, prior_std = torch.chunk(self.prior_fc(deter_state), 2, dim=-1)

        # Posterior
        post_mean, post_std = torch.chunk(self.post_fc(deter_state), 2, dim=-1)
        stoch_state = post_mean + torch.randn_like(post_std) * post_std

        # Reward prediction
        feat = torch.cat([deter_state, stoch_state], dim=-1)
        reward_pred = self.reward_net(feat)

        return (deter_state, stoch_state), reward_pred, (prior_mean, prior_std)


class Dreamer(nn.Module):
    def __init__(self, config):
        super().__init__()
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

    def imagine_rollout(self, initial_state, horizon):
        states = []
        actions = []
        rewards = []
        current_state = initial_state

        for _ in range(horizon):
            action = self.actor(torch.cat(current_state, dim=-1))
            next_state, reward_pred, _ = self.rssm(None, action, current_state)

            states.append(current_state)
            actions.append(action)
            rewards.append(reward_pred)
            current_state = next_state

        return states, actions, rewards
