import torch
import torch.nn as nn
from typing import Tuple

class PPOAgent(nn.Module):
    """Proximal Policy Optimization Agent."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_size: int = 128
    ) -> None:
        """Initialize the PPOAgent.

        Args:
            observation_dim (int): Dimension of the observation space.
            action_dim (int): Dimension of the action space.
            hidden_size (int, optional): Size of the hidden layers. Defaults to 128.
        """
        super(PPOAgent, self).__init__()
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh(),
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        observation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the policy and value networks.

        Args:
            observation (torch.Tensor): The input observation tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The action mean and the value estimate.
        """
        action_mean = self.policy_net(observation)
        value = self.value_net(observation).squeeze(-1)
        return action_mean, value

    def act(
        self,
        observation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select an action based on the current policy.

        Args:
            observation (torch.Tensor): The input observation tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                The sampled action, log probability, and value estimate.
        """
        action_mean = self.policy_net(observation)
        action_std = torch.ones_like(action_mean) * 0.1
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.value_net(observation).squeeze(-1)
        return action, log_prob, value
