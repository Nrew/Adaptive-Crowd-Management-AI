import torch
import torch.nn as nn
from typing import Tuple
from dataclasses import dataclass

@dataclass(frozen=True)
class AgentProfile:
    """
    Immutable agent personaility configuration
    """
    base_panic_threshold: float # Base threshold for panic behvaior
    social_influence: float # How much an agent is influenced by others (0-1)
    helping_tendency: float # Probablity of helping others
    decision_speed: float # How quickly agent makes decisions
    risk_aversion: float # Tendency to avoid dangerous situations
    emotional_stability: float # Resistance to panic (0-1)
    leadership_score: float # Influence on other agents (0-1)


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
            nn.ReLU,
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)
    
    def act(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Select an action based on the policy network's output.

        Args:
            state (torch.Tensor): The state to act on.

        Returns:
            Tuple[int, torch.Tensor]: Selected action and the log probability.
        """
        action_probs = self.forward(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)