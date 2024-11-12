import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple

from ai_models.ppo import MovementObservation

class PolicyNetwork(nn.Module):
    def __init__(self, config: Dict):
        super(PolicyNetwork, self).__init__()
        self.hidden_size = config['policy_network']['hidden_size']
        self.num_attention_heads = config['policy_network']['num_attention_heads']

        # Define processing layers for different parts of the observation
        self.position_processor = nn.Sequential(
            nn.Linear(2, self.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size)
        )

        self.visible_agents_processor = nn.Sequential(
            nn.Linear(5 + 3, self.hidden_size),  # Agent data + emotions
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size)
        )

        self.hazards_processor = nn.Sequential(
            nn.Linear(3, self.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size)
        )

        self.exits_processor = nn.Sequential(
            nn.Linear(2, self.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size)
        )

        self.walls_processor = nn.Sequential(
            nn.Linear(2, self.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size)
        )

        # Local map processor (e.g., using convolutional layers)
        self.local_map_processor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * config['grid_size'] * config['grid_size'], self.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size)
        )

        # Attention mechanism for visible agents
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_attention_heads,
            batch_first=True
        )

        # Final policy head
        self.policy_head = nn.Sequential(
            nn.Linear(self.hidden_size * 5, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, config['action_dim'])  # action_dim depends on your action space
        )

    def forward(self, observation: MovementObservation) -> torch.Tensor:
        batch_size = observation.position.size(0)

        # Process agent's position
        position_features = self.position_processor(observation.position)  # [batch_size, hidden_size]

        # Process visible agents
        if observation.visible_agents.size(1) > 0:
            # Concatenate visible agents' data and emotions
            visible_agents_data = torch.cat([observation.visible_agents, observation.visible_emotions], dim=-1)
            visible_agents_features = self.visible_agents_processor(
                visible_agents_data.view(-1, 8)
            ).view(batch_size, -1, self.hidden_size)

            # Apply attention mechanism
            attn_output, _ = self.attention(
                query=position_features.unsqueeze(1),  # [batch_size, 1, hidden_size]
                key=visible_agents_features,           # [batch_size, N, hidden_size]
                value=visible_agents_features,         # [batch_size, N, hidden_size]
                key_padding_mask=~observation.agent_mask  # [batch_size, N]
            )
            social_features = attn_output.squeeze(1)  # [batch_size, hidden_size]
        else:
            social_features = torch.zeros(batch_size, self.hidden_size, device=position_features.device)

        # Process hazards
        if observation.hazards.size(1) > 0:
            hazards_features = self.hazards_processor(
                observation.hazards.view(-1, 3)
            ).view(batch_size, -1, self.hidden_size)
            hazards_features = hazards_features.mean(dim=1)
        else:
            hazards_features = torch.zeros(batch_size, self.hidden_size, device=position_features.device)

        # Process exits
        if observation.exits.size(1) > 0:
            exits_features = self.exits_processor(
                observation.exits.view(-1, 2)
            ).view(batch_size, -1, self.hidden_size)
            exits_features = exits_features.mean(dim=1)
        else:
            exits_features = torch.zeros(batch_size, self.hidden_size, device=position_features.device)

        # Process walls
        if observation.walls.size(1) > 0:
            walls_features = self.walls_processor(
                observation.walls.view(-1, 2)
            ).view(batch_size, -1, self.hidden_size)
            walls_features = walls_features.mean(dim=1)
        else:
            walls_features = torch.zeros(batch_size, self.hidden_size, device=position_features.device)

        # Process local map
        local_map_features = self.local_map_processor(
            observation.local_map.unsqueeze(1)  # Add channel dimension
        )  # [batch_size, hidden_size]

        # Combine features
        combined_features = torch.cat(
            [position_features, social_features, hazards_features, exits_features, local_map_features], dim=-1
        )

        # Output action logits
        action_logits = self.policy_head(combined_features)
        return action_logits
