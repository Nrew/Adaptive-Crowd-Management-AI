from typing import Dict, Tuple, Optional, NamedTuple
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass


class MovementObservation(NamedTuple):
    """Structured observation from environment."""
    position: torch.Tensor              # Agent's position [2]
    visible_agents: torch.Tensor        # Other agents' data [N, 5] (pos[2], vel[2], distance[1])
    visible_emotions: torch.Tensor      # Other agents' emotions [N, 3]
    hazards: torch.Tensor               # Hazard positions and intensities [M, 3]
    exits: torch.Tensor                 # Exit positions [K, 2]
    walls: torch.Tensor                 # Wall/obstacle positions [W, 2]
    agent_mask: torch.Tensor            # Mask for valid agents [N]
    local_map: torch.Tensor             # Local area grid encoding [grid_size, grid_size]

@dataclass
class MovementDecision:
    """Container for movement decision outputs."""
    action: torch.Tensor               # Selected action
    action_log_prob: torch.Tensor      # Log probability of action
    value: torch.Tensor                # State value estimate
    hidden_state: Optional[torch.Tensor] # RNN hidden state if used

class FeatureExtractor(nn.Module):
    """Extracts features from observations considering the emotional states."""
    def __init__(
        self,
        config: Dict,
        hidden_size: int = 256
    ):
        super().__init__()
        
        # Process agent's own state and emotions
        self.self_encoder = nn.Sequential(
            nn.Linear(5, hidden_size),  # position(2) + emotion(3)
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        
        # Process visible agents
        self.agent_encoder = nn.Sequential(
            nn.Linear(8, hidden_size),  # pos(2) + vel(2) + distance(1) + emotion(3)
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        
        # Attention for agent interactions
        self.agent_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            batch_first=True
        )
        
        # Process hazards and exits
        self.hazard_encoder = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        
        self.exit_encoder = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        
        # Process local map
        self.map_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, hidden_size)
        )
        
        # Combine all features
        self.feature_combiner = nn.Sequential(
            nn.Linear(hidden_size * 5, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
    
    def forward(
        self,
        obs: MovementObservation,
        emotional_state: torch.Tensor
    ) -> torch.Tensor:
        """Extract features from observation."""
        # Combine position and emotional state
        self_state = torch.cat([
            obs.position,
            emotional_state
        ])
        self_features = self.self_encoder(self_state)
        
        # Process visible agents with attention
        if obs.visible_agents.size(0) > 0:
            agent_states = torch.cat([
                obs.visible_agents,
                obs.visible_emotions
            ], dim=-1)
            agent_features = self.agent_encoder(agent_states)
            
            # Apply attention
            agent_features, _ = self.agent_attention(
                agent_features.unsqueeze(0),
                agent_features.unsqueeze(0),
                agent_features.unsqueeze(0),
                key_padding_mask=~obs.agent_mask.unsqueeze(0)
            )
            agent_features = agent_features.squeeze(0).mean(0)
        else:
            agent_features = torch.zeros_like(self_features)
        
        # Process hazards and exits
        hazard_features = self.hazard_encoder(obs.hazards).mean(0) \
            if obs.hazards.size(0) > 0 else torch.zeros_like(self_features)
        
        exit_features = self.exit_encoder(obs.exits).mean(0) \
            if obs.exits.size(0) > 0 else torch.zeros_like(self_features)
        
        # Process local map
        map_features = self.map_encoder(obs.local_map.unsqueeze(0).unsqueeze(0))
        
        # Combine all features
        combined = torch.cat([
            self_features,
            agent_features,
            hazard_features,
            exit_features,
            map_features
        ])
        
        return self.feature_combiner(combined)

class PPO:
    def __init__(self,
                 policy_network: nn.Module,
                 value_network: nn.Module,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 clip_epsilon: float = 0.2) -> None:
        """
        Proximal Policy Optimization Algorithm.

        Args:
            policy_network (nn.Module): The policy network (actor).
            value_network (nn.Module): The value network (critic)
            lr (float, optional): Learning rate. Defaults to 3e-4.
            gamma (float, optional): Discount factor for rewards. Defaults to 0.99.
            clip_epsilon (float, optional): Clipping parameter for PPO. Defaults to 0.2.
        """
        self.policy_network = policy_network
        self.value_network = value_network
        self.optimizer = optim.Adam(
            list(self.policy_network.parameters()) + list(self.value_network.parameters()),
            lr=lr
        )
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
    def __compute_loss(self,
                     states: torch.Tensor,
                     actions: torch.Tensor,
                     log_prob_old: torch.Tensor,
                     rewards: torch.Tensor,
                     adavntages: torch.Tensor) -> torch.Tensor:
        """
        Computes the PPO loss.

        Args:
            states (torch.Tensor): _description_
            actions (torch.Tensor): _description_
            log_prob_old (torch.Tensor): _description_
            rewards (torch.Tensor): _description_
            adavntages (torch.Tensor): _description_

        Returns:
            torch.Tensor: Combined policy and value loss.
        """
        log_prob_new = self.policy_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        ratio = torch.exp(log_prob_new - log_prob_old)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        policy_loss = -torch.min(ratio * adavntages, clipped_ratio * adavntages).mean()
        
        value_estimates = self.value_network(states).squeeze(1)
        value_loss = nn.MSELoss()(value_estimates, rewards)
        
        return policy_loss + 0.5 * value_loss
    def update(self,
                     states: torch.Tensor,
                     actions: torch.Tensor,
                     log_prob_old: torch.Tensor,
                     rewards: torch.Tensor,
                     adavntages: torch.Tensor) -> None:
        """
        Update the policy and value networks based on PPO loss.

        Args:
            states (torch.Tensor): _description_
            actions (torch.Tensor): _description_
            log_prob_old (torch.Tensor): _description_
            rewards (torch.Tensor): _description_
            adavntages (torch.Tensor): _description_
        """
        loss = self.__compute_loss(states, actions, log_prob_old, rewards, adavntages)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()