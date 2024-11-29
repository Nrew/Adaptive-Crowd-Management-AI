# ppo_agent.py

import torch
import torch.nn as nn

class PPOAgent(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_size=128):
        super(PPOAgent, self).__init__()
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, observation):
        action_mean = self.policy_net(observation)
        value = self.value_net(observation).squeeze(-1)
        return action_mean, value

    def act(self, observation):
        action_mean = self.policy_net(observation)
        action_std = torch.ones_like(action_mean) * 0.1
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.value_net(observation).squeeze(-1)
        return action, log_prob, value
