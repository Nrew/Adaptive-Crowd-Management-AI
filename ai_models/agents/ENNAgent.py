import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .EmotionalState import EmotionalState
from models.ENN import EmotionalNetwork


class PPOAgentWithENN(nn.Module):
    """Proximal Policy Optimization Agent with Emotional Neural Network."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        enn_config: Optional[dict] = None
    ) -> None:
        """Initialize the PPOAgentWithENN.

        Args:
            observation_dim (int): Dimension of the observation space.
            action_dim (int): Dimension of the action space.
            hidden_size (int, optional): Size of the hidden layers. Defaults to 128.
            enn_config (Optional[dict], optional): Configuration for the Emotional Network.
                Defaults to None.
        """
        super(PPOAgentWithENN, self).__init__()

        self.enn = EmotionalNetwork(enn_config)
        self.emotional_state = EmotionalState.create_initial().to_tensor(
            torch.device('cpu')
        ).unsqueeze(0)

        # Feature extractors
        self.obs_feature_extractor = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.emotion_feature_extractor = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Attention mechanism for dynamic emotion influence
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),
            nn.Sigmoid()
        )

        # Combined network
        self.combined_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Policy network head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()
        )

        # Value network head
        self.value_head = nn.Linear(hidden_size, 1)

        # Urgency reward scaling
        self.urgency_scaling = 1.0  # Can be adjusted dynamically

    def forward(
        self,
        observation: torch.Tensor,
        nearby_agents: Optional[torch.Tensor] = None,
        nearby_hazards: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the policy and value networks.

        Args:
            observation (torch.Tensor): The input observation tensor.
            nearby_agents (Optional[torch.Tensor], optional): Information about nearby agents.
                Defaults to None.
            nearby_hazards (Optional[torch.Tensor], optional): Information about nearby hazards.
                Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The action mean and the value estimate.
        """
        # Update emotional state using ENN
        self.emotional_state = self.enn(
            self.emotional_state,
            observation.unsqueeze(0),
            nearby_agents,
            nearby_hazards
        )

        # Normalize emotional state for stability
        normalized_emotion_state = F.normalize(self.emotional_state.squeeze(0), p=2, dim=-1)

        # Feature extraction
        obs_features = self.obs_feature_extractor(observation)
        emotion_features = self.emotion_feature_extractor(normalized_emotion_state)

        # Dynamic attention mechanism
        attention_weights = self.attention_layer(torch.cat([obs_features, emotion_features], dim=-1))
        combined_features = attention_weights * obs_features + (1 - attention_weights) * emotion_features

        # Pass through the combined network
        combined_output = self.combined_net(combined_features)

        # Policy and value outputs
        action_mean = self.policy_head(combined_output)
        value = self.value_head(combined_output).squeeze(-1)

        return action_mean, value

    def act(
        self,
        observation: torch.Tensor,
        nearby_agents: Optional[torch.Tensor] = None,
        nearby_hazards: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select an action based on the current policy.

        Args:
            observation (torch.Tensor): The input observation tensor.
            nearby_agents (Optional[torch.Tensor], optional): Information about nearby agents.
                Defaults to None.
            nearby_hazards (Optional[torch.Tensor], optional): Information about nearby hazards.
                Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                The sampled action, log probability, and value estimate.
        """
        action_mean, value = self.forward(
            observation,
            nearby_agents,
            nearby_hazards
        )
        action_std = torch.ones_like(action_mean) * 0.1  # Can be adjusted based on urgency
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        # Urgency adjustment to log probability (encourages faster decision-making)
        log_prob *= self.urgency_scaling

        return action, log_prob, value