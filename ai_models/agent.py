import torch
import torch.nn as nn
from typing import Protocol, Tuple
from dataclasses import dataclass

class AgentState(Protocol):
    """Protocol defining required agent state interface."""
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    panic_level: float

class Agent(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        """
        RL Agent using a neural network for the policy

        Args:
            input_dim (int): Dimentionality of input.
            output_dim (int): Number of possible actions.
        """
        super(Agent, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)
    
    def act(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Select continuous actions based on the policy network's output.

        Args:
            state (torch.Tensor): The state to act on.

        Returns:
            Tuple[int, torch.Tensor]: Selected action and the log probability.
        """
        # print(f"Here is the state: {state} \n After forward: {self.forward(state)}")
        action_mean = self.forward(state)
        action_std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(action_mean, action_std)
        # print(f"Action distribution: {dist, dist.loc, dist.scale}")
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        return action, log_prob