import time
import torch.nn.functional as F
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, NamedTuple, TypeVar

from .processor import FeatureProcessor

@dataclass(slots=True) 
class SafetyBounds:
    """
    Dataclass for safety bounds for emotional transitions.
    """
    max_panic_increase:     float = 0.3
    max_stress_increase:    float = 0.2
    min_stamina:            float = 0.1
    max_social_influence:   float = 0.5 
    max_hazard_impact:      float = 0.4
    
    
    def get_scaled_bounds(
        self,
        risk_factor: torch.Tensor,
        stability: torch.Tensor
    ) -> torch.Tensor:
        """Get bounds scaled by agent profile."""
        base_bounds = torch.tensor([
            self.max_panic_increase,
            self.max_stress_increase,
            1.0 - self.min_stamina
        ], device=risk_factor.device)
        
        # Scale based on risk aversion and stability
        scale = torch.lerp(
            torch.ones_like(risk_factor),
            0.5 * torch.ones_like(risk_factor),
            risk_factor
        ) * (1.0 - 0.5 * stability)
        
        return base_bounds * scale
    
class EmotionalActivations(NamedTuple):
    """
    Container for network activations
    """
    hazard_features:        torch.Tensor
    social_features:        torch.Tensor
    attention_weights:      torch.Tensor
    final_features:         torch.Tensor
    
class FeatureCache:
    """Cache for computed features with automatic expiration.
        
        Consider 100 agents near the same hazard
        hazard = torch.tensor([1.0, 1.0, 0.9])  # Same hazard position and intensity
            
        Without cache:
            Processes hazard 100 times (once per agent)
            O(100 * processing_time)

        With cache:
            First agent: Cache miss, processes hazard
            Next 99 agents: Cache hits, instant retrieval
            O(1 * processing_time + 99 * retrieval_time)
        
    """
    
    def __init__(self, capacity: int = 1000, ttl: float = 0.1):
        self.capacity = capacity
        self.ttl = ttl
        self.cache = {}
        self.timestamps = {}
        self.hits = 0
        self.misses = 0
    
    def get(self, key: torch.Tensor) -> Optional[torch.Tensor]:
        """Get cached feature if available and not expired."""
        key_hash = hash(key.cpu().numpy().tobytes())
        
        if key_hash in self.cache:
            if time.time() - self.timestamps[key_hash] < self.ttl:
                self.hits += 1
                return self.cache[key_hash]
            else:
                del self.cache[key_hash]
                del self.timestamps[key_hash]
        
        self.misses += 1
        return None
    
    def put(self, key: torch.Tensor, value: torch.Tensor):
        """Cache computed feature."""
        key_hash = hash(key.cpu().numpy().tobytes())
        
        if len(self.cache) >= self.capacity:
            # Remove oldest entry
            oldest_key = min(self.timestamps, key=self.timestamps.get)
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
        
        self.cache[key_hash] = value
        self.timestamps[key_hash] = time.time()
    
class AttentionModule(nn.Module):
    """Multi-head attention block for social influence."""
    def __init__(
        self, 
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        batch_first: bool = True
    ) -> None:
        
        super(AttentionModule, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first
        )
        
        self.use_flash = hasattr(F, 'scaled_dot_product_attention')
        if self.use_flash:
            # Flash attention specific initializations
            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(embed_dim, embed_dim)
            self.v_proj = nn.Linear(embed_dim, embed_dim)
            self.o_proj = nn.Linear(embed_dim, embed_dim)
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None    
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with weight tracking."""
        if not self.use_flash:
            # Fallback to regular attention
            return self.regular_attention(
                query, key, value,
                key_padding_mask=mask,
                need_weights=True
            )
        
        batch_size = query.size(0)
            
        # Project inputs
        q = self.q_proj(query).view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(key).view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(value).view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        # Apply Flash Attention
        attention_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )
        
        # Reshape and project output
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.embed_dim
        )
        output = self.o_proj(attention_output)
        
        # No attention weights returned for Flash Attention
        return output, None

class EmotionalPredictor(FeatureProcessor):
    """
    Predicts new emotional states.
    """
    def __init__(self, hidden_dim, dropout = 0.1):
        super(EmotionalPredictor, self).__init__(
            input_dim=hidden_dim * 3, # Combined features
            hidden_dim=hidden_dim,
            output_dim=3,             # [panic, stress, stamina]
            dropout=dropout)

class HazardProcessor(FeatureProcessor):
    """
    Process hazard information
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super(HazardProcessor, self).__init__(
            input_dim=3,            # Position: [X, Y] + Intensity: (1)
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout)

class SocialProcessor(FeatureProcessor):
    """
    Processes social influence information.
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super(SocialProcessor, self).__init__(
            input_dim=10,            # EmotionalState (3) + Profile (7)
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout)

class EmotionalNetwork(nn.Module):
    """
    Neural network for processing emotional dynamics.
    
    This network processes agent emotional states, considering:
    - Individual agent profiles
    - Social influences from nearby agents
    - Enviromental hazards
    - Safety bounds and constraints
    """
    
    def __init__(self, config: Dict):
        """
        Initalize emotional netowrk.

        Args:
            config (Dict): Configuration dictionary containing:
                - network parameters ( hidden_size, num_heads, etc.)
                - safety bounds
                - maximum agents/hazards
                - debug settings
        """
        super(EmotionalNetwork, self).__init__()
        
        hidden_size = config['emotional']['network']['hidden_size']
        num_heads = config['emotional']['network']['num_attention_heads']
        dropout = config['emotional']['network']['dropout']
        
        self.hazard_processor = HazardProcessor(
            hidden_dim=hidden_size,
            dropout=dropout
        )
        
        self.social_processor = SocialProcessor(
            hidden_dim=hidden_size,
            dropout=dropout
        )
        
        self.attention = AttentionModule(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.predictor = EmotionalPredictor(
            hidden_dim=hidden_size,
            dropout=dropout
        )
        
        # Feature cache
        self.hazard_cache = FeatureCache(
            capacity=config['cache']['hazard_capacity'],
            ttl=config['cache']['ttl']
        )
        self.social_cache = FeatureCache(
            capacity=config['cache']['social_capacity'],
            ttl=config['cache']['ttl']
        )
               
        self.safety_bounds = SafetyBounds(**config['safety_bounds'])
        
        self.register_buffer('stats', torch.zeros(4))  # [hits, misses, time, count]
        
    def _process_hazards(
        self,
        hazards: torch.Tensor,  # [batch_size, num_hazards, 3]
        hazard_mask: Optional[torch.Tensor] = None  # [batch_size, num_hazards]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process hazard information with caching and masking.
        Returns processed features and hazard influence.
        """
        cached_features = self.hazard_cache.get(hazards)
        if cached_features is not None:
            return cached_features
            
        # Calculate if not cached
        features, influence = self._calculate_hazard_features(hazards, hazard_mask)
        self.hazard_cache.put(hazards, (features, influence))
        return features, influence

    def _process_social(
        self,
        states: torch.Tensor,          # [batch_size, num_agents, 3]
        profiles: torch.Tensor,        # [batch_size, num_agents, 7]
        distances: torch.Tensor,       # [batch_size, num_agents]
        valid_mask: torch.Tensor,      # [batch_size, num_agents]
        max_distance: float = 10.0     # Maximum influence distance
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process social information with distance-based weighting.
        Returns processed features and social influence weights.
        """
        batch_size, num_agents = states.size()[:2]
        
        # Calculate distance-based weights
        distance_weights = torch.clamp(
            1.0 - (distances / max_distance),
            min=0.0
        ).unsqueeze(-1)  # [batch_size, num_agents, 1]
        
        # Apply valid mask
        masked_weights = distance_weights * valid_mask.unsqueeze(-1)
        
        # Combine features with distance weighting
        combined = torch.cat([
            states,           # emotional states
            profiles,        # agent profiles
            masked_weights   # distance weights
        ], dim=-1)
        
        # Process through network
        social_features = self.social_processor(combined)
        
        # Calculate weighted influence
        leadership_scores = profiles[..., 6:7]  # Extract leadership scores
        influence_weights = masked_weights * leadership_scores
        
        # Normalize weights
        influence_weights = influence_weights / (
            torch.sum(influence_weights, dim=1, keepdim=True) + 1e-6
        )
        
        return social_features, influence_weights

    def forward(
        self,
        current_state: torch.Tensor,
        agent_profile: torch.Tensor,
        nearby_agents: Tuple[torch.Tensor, torch.Tensor],
        nearby_hazards: torch.Tensor,
        hazard_mask: Optional[torch.Tensor] = None,
        delta_time: float = 0.1
    ) -> torch.Tensor:
        """Forward pass using JIT-compiled processing methods."""
        # Process hazards
        hazard_features, hazard_influence = self._process_hazards(
            nearby_hazards,
            hazard_mask
        )
        
        # Calculate distances for social processing
        nearby_states, nearby_profiles = nearby_agents
        agent_positions = nearby_states[..., :2]
        distances = torch.norm(
            agent_positions.unsqueeze(2) - agent_positions.unsqueeze(1),
            dim=-1
        )
        
        # Create valid mask (exclude self-interactions)
        valid_mask = torch.ones_like(distances, dtype=torch.bool)
        valid_mask.diagonal(dim1=1, dim2=2).fill_(False)
        
        # Process social features
        social_features, social_weights = self._jit_process_social(
            nearby_states,
            nearby_profiles,
            distances,
            valid_mask
        )
        
        # Combine features
        combined_features = torch.cat([
            hazard_features * hazard_influence.unsqueeze(-1),
            social_features * social_weights
        ], dim=-1)
        
        # Final prediction
        new_state = self.predictor(combined_features)
        
        return new_state
