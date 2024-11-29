import torch
import torch.nn as nn
import torch.nn.functional as F

from ai_models.agents import EmotionalState
from ai_models.models.ENN import EmotionalNetwork

class PPOAgentWithENN(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_size=128, enn_config=None):
        super(PPOAgentWithENN, self).__init__()
        self.enn = EmotionalNetwork(enn_config)
        self.emotional_state = EmotionalState.create_initial().to_tensor(torch.device('cpu')).unsqueeze(0)

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

    def forward(self, observation, nearby_agents=None, nearby_hazards=None):
        # Update emotional state using ENN
        self.emotional_state = self.enn(
            self.emotional_state,
            observation.unsqueeze(0),  # Adjust dimensions as needed
            nearby_agents,
            nearby_hazards
        )

        # Feature extraction
        obs_features = self.obs_feature_extractor(observation)
        emotion_features = self.emotion_feature_extractor(self.emotional_state.squeeze(0))

        # Combine features
        combined_features = torch.cat([obs_features, emotion_features], dim=-1)
        combined_output = self.combined_net(combined_features)

        # Policy and value outputs
        action_mean = self.policy_head(combined_output)
        value = self.value_head(combined_output).squeeze(-1)

        return action_mean, value

    def act(self, observation, nearby_agents=None, nearby_hazards=None):
        action_mean, value = self.forward(observation, nearby_agents, nearby_hazards)
        action_std = torch.ones_like(action_mean) * 0.1
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value
