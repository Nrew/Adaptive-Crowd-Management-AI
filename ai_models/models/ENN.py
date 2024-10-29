import torch
import numpy as np
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Protocol, Tuple, Dict, Optional, NamedTuple

from ai_models.agents.profile import AgentProfile


class EmotionalStateProtocol(Protocol):
    """
    Protocol defining interface for emotional states.
    """
    
    @property
    def panic(self) -> float:
        """
        Get panic level.

        Returns:
            float: _description_
        """
        ...
    @property
    def stress(self) -> float:
        """
        Get stress level.

        Returns:
            float: _description_
        """
        ...
    @property
    def stamina(self) -> float:
        """
        Get staminal level.

        Returns:
            float: _description_
        """
        ...
    def to_tensor(self, device: torch.device) -> torch.Tensor:
        """
        Convert state to tensor.

        Args:
            device (torch.device): _description_

        Returns:
            torch.Tensor: _description_
        """
        ...

@dataclass
class EmotionalState:
    """
    Dataclass struct for an Agent's emotional state.
    """
    panic: float    # Current panic level [0-1]
    stress: float   # Current stress level [0-1]
    stamina: float  # Current stamina level [0-1]
    
    @classmethod
    def create_initial(cls) -> 'EmotionalState':
        """
        Create initial emotional state.

        Returns:
            EmotionalState: An instance of an emotional state with the predefined data.
        """
        return cls(
            panic = 0.0,
            stress = 0.0,
            stamina = 1.0
        )
    
    def to_tensor(self, device: torch.device) -> torch.Tensor:
        """
        Convert emotional state to a tensor.

        Args:
            device (torch.device): GPU or CPU

        Returns:
            torch.Tensor: EmotionalState as a pytorch tensor.
        """
        return torch.tensor(
            [self.panic, self.stress, self.stamina],
            dtype=torch.float32,
            device=device
        )

@dataclass
class SafetyBounds:
    """
    Dataclass for safety bounds for emotional transitions.
    """
    max_panic_increase: float = 0.3
    max_stress_increase: float = 0.2
    min_stamina: float = 0.1
    max_social_infulence: float = 0.5
    max_hazard_impact: float = 0.4
    
class EmotionalActivations(NamedTuple):
    """
    Container for network activations
    """
    hazard_features: torch.Tensor
    social_features: torch.Tensor
    attention_weights: torch.Tensor
    final_features: torch.Tensor
    
class AttentionModule(nn.Module):
    """Multi-head attention block for socual influence."""
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1) -> None:
        super(AttentionModule, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.last_attention_weights: Optional[torch.Tensor] = None
    
    def forward(self,
                queries: torch.Tensor,
                keys: torch.Tensor,
                values: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with weight tracking."""
        attended, weights = self.attention(
            queries, keys, values,
            key_padding_mask=mask,
            need_weights=True
        )
        
        self.last_attention_weights = weights
        attended = self.dropout(attended)
        return self.layer_norm(queries + attended)

class FeatureProcessor(nn.Module):
    """
    Base class for feature processing modules.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 dropout: float = 0.1) -> None:
        super(FeatureProcessor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process features through network.
        """
        return self.network(x)

class HazardProcessor(FeatureProcessor):
    """
    Process hazard information
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super(HazardProcessor, self).__init__(
            input_dim=3, # Position: [X, Y] + Intensity: (1)
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout)

class SocialProcessor(FeatureProcessor):
    """
    Processes social influence information.
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super(SocialProcessor, self).__init__(
            input_dim=10, # EmotionalState (3) + Profile (7)
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout)

class EmotionalPredictor(FeatureProcessor):
    """
    Predicts new emotional states.
    """
    def __init__(self, hidden_dim, dropout = 0.1):
        super(EmotionalPredictor, self).__init__(
            input_dim=hidden_dim * 3, # Combined features
            hidden_dim=hidden_dim,
            output_dim=3, # [panic, stress, stamina]
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
        
        # Cache tensor sizes for pre-allocation
        self.max_agents = config['enviroment']['max_nearby_agents']
        self.max_hazards = config['enviroment']['max_hazards']
        
        
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
        
        self.safety_bounds = SafetyBounds(
            **config.get('safety_bounds', {})
        )
        
        # Debug mode
        self.debug_mode = False
        self.activation_history: List[EmotionalActivations] = []
        
        # Register hooks for analysis
        self.__register_hooks()
        
        # Initialize Buffers
        self.__init_buffers()
        
        # JIT compile key methods
        self.__compile_methods()
        
        
    def __init_buffers(self):
        """
        Iniitalize reuseable buffers
        """
        self.register_buffer(
            'attention_mask',
            torch.ones(1, self.max_agents)
        )
        self.register_buffer(
            'agent_indicies',
            torch.arange(self.max_agents)
        )
        self.register_buffer(
            'distance_scale',
            torch.tensor([1.0, 1.0, 0.5]) # xy- distance, intensity scale, 
        )
        
    
    def __register_hooks(self):
        """
        Register hooks for activation tracking.
        """
        self.activations = {}
        
        def hook_fn(name: str):
            def hook(module, input, output):
                if self.debug_mode:
                    self.activations[name] = output.detach()
            return hook
        
        self.hazard_processor.register_forward_hook(hook_fn('hazard'))
        self.attention.register_forward_hook(hook_fn('attention'))
        self.predictor.register_forward_hook(hook_fn('predictor'))
        
    def __compile_methods(self):
        """
        JIT compile performance critical methods.
        """
        self.process_batch = torch.jit.script(self._process_batch)
        self.process_attention = torch.jit.script(self._process_attention)
    @torch.jit.script_method
    def _process_batch(
        self,
        emotional_states: torch.Tensor,  # [batch_size, 3]
        agent_profiles: torch.Tensor,    # [batch_size, 7]
        context: torch.Tensor            # [batch_size, 3]
    ) -> torch.Tensor:
        """Process batch of agent data."""
        features = torch.cat([emotional_states, agent_profiles, context], dim=-1)
        return self.social_processor(features)
    
    @torch.jit.script_method
    def _process_attention(
        self,
        features: torch.Tensor,         # [batch_size, hidden_size]
        nearby_features: torch.Tensor,  # [batch_size, max_agents, hidden_size]
        mask: torch.Tensor              # [batch_size, max_agents]
    ) -> torch.Tensor:
        """Apply attention mechanism."""
        query = features.unsqueeze(1)
        attended = self.attention(query, nearby_features, nearby_features, ~mask)
        return attended.squeeze(1)
    
    def forward(
        self,
        current_state: torch.Tensor,      # [batch_size, 3]
        agent_profile: torch.Tensor,      # [batch_size, 7]
        nearby_agents: Tuple[torch.Tensor, torch.Tensor],  # ([batch_size, max_agents, 3], [batch_size, max_agents, 7])
        nearby_hazards: torch.Tensor,     # [batch_size, max_hazards, 3]
        delta_time: float
    ) -> torch.Tensor:
        """Process emotional dynamics for a batch of agents.
        
        Args:
            current_state: Current emotional states
            agent_profile: Agent personality profiles
            nearby_agents: Tuple of (emotional_states, profiles) for nearby agents
            nearby_hazards: Hazard information (position and intensity)
            delta_time: Time step size
        
        Returns:
            torch.Tensor: Updated emotional states
        """
        batch_size = current_state.size(0)
        device = current_state.device
        
        # Process hazards
        hazard_features = self.hazard_processor(nearby_hazards)
        hazard_influence = self._calculate_hazard_influence(
            hazard_features,
            nearby_hazards
        )
        
        # Process social influence
        nearby_emotions, nearby_profiles = nearby_agents
        valid_mask = (nearby_emotions.abs().sum(-1) > 0)
        
        nearby_features = self._process_batch(
            nearby_emotions.view(-1, 3),
            nearby_profiles.view(-1, 7),
            hazard_influence.repeat_interleave(self.max_agents, 0)
        ).view(batch_size, self.max_agents, -1)
        
        # Process current agent
        agent_features = self._process_batch(
            current_state,
            agent_profile,
            hazard_influence
        )
        
        # Apply social attention
        social_features = self._process_attention(
            agent_features,
            nearby_features,
            valid_mask
        )
        
        # Predict new emotional state
        combined = torch.cat([
            agent_features,
            social_features,
            hazard_features.mean(1)
        ], dim=-1)
        
        new_state = self.predictor(combined)
        
        # Apply safety bounds and time-based decay
        new_state = self._apply_safety_bounds(
            current_state,
            new_state,
            agent_profile,
            delta_time
        )
        
        # Store debug information if needed
        if self.debug_mode:
            self._store_debug_info(
                hazard_features,
                social_features,
                combined
            )
        
        return new_state
    
    def _calculate_hazard_influence(
        self,
        hazard_features: torch.Tensor,
        hazard_data: torch.Tensor
    ) -> torch.Tensor:
        """Calculate hazard influence with distance scaling."""
        distances = torch.norm(hazard_data[..., :2], dim=-1, keepdim=True)
        intensities = hazard_data[..., 2:3]
        
        # Apply inverse square law with safety bound
        influence = (intensities / (1 + distances.pow(2)))
        influence = influence.clamp(
            max=self.safety_bounds.max_hazard_impact
        )
        
        return influence * hazard_features
    
    def _apply_safety_bounds(
        self,
        current_state: torch.Tensor,
        new_state: torch.Tensor,
        agent_profile: torch.Tensor,
        delta_time: float
    ) -> torch.Tensor:
        """Apply safety bounds to emotional transitions."""
        # Calculate maximum allowed changes
        max_changes = torch.tensor([
            self.safety_bounds.max_panic_increase,
            self.safety_bounds.max_stress_increase,
            1.0 - self.safety_bounds.min_stamina
        ], device=current_state.device)
        
        # Scale by time step and emotional stability
        stability = agent_profile[:, 5:6]  # emotional_stability
        max_changes = max_changes * delta_time * (1 - stability)
        
        # Clamp changes
        state_change = new_state - current_state
        clamped_change = torch.clamp(
            state_change,
            min=-max_changes,
            max=max_changes
        )
        
        # Apply clamped changes
        bounded_state = current_state + clamped_change
        
        # Ensure stamina stays above minimum
        bounded_state[:, 2] = torch.clamp(
            bounded_state[:, 2],
            min=self.safety_bounds.min_stamina,
            max=1.0
        )
        
        return bounded_state
    
    def _store_debug_info(
        self,
        hazard_features: torch.Tensor,
        social_features: torch.Tensor,
        combined_features: torch.Tensor
    ):
        """Store activation information for debugging."""
        self.activation_history.append(
            EmotionalActivations(
                hazard_features=hazard_features.detach(),
                social_features=social_features.detach(),
                attention_weights=self.attention.last_attention_weights.detach(),
                final_features=combined_features.detach()
            )
        )
    
    def reset_debug(self):
        """Reset debug information."""
        self.activation_history.clear()
        self.activations.clear()


@DeprecationWarning
class EmotionalDynamics:
    """
    Handles emotional state updates and social contagion.
    """
    
    def __init__(self, config: Dict) -> None:
        self.panic_threshold = config['emotional']['panic_threshold']
        self.social_influence_radius = config['emotional']['social_influence_radius']
        self.emotional_decay_rate = config['emotional']['emotional_decay_rate']
        self.stress_accumulation_rate = config['emotional']['stress_accumlation_rate']
    
    def update_emotional_state(self,
                               current_state: EmotionalState,
                               nearby_agents_states: List[EmotionalState],
                               nearby_hazards: torch.Tensor,
                               delta_time: float) -> EmotionalState:
        """
        Update agent's emotional state based on environment and social factors.

        Args:
            current_state (EmotionalState): _description_
            nearby_agents_states (List[EmotionalState]): _description_
            nearby_hazards (torch.Tensor): _description_
            delta_time (float): _description_

        Returns:
            EmotionalState: _description_
        """
        
        # Environmental influence
        hazard_intensity = self.__calculate_hazard_influence(nearby_hazards)
        
        # Social Contagion
        social_panic = self.__calculate_social_panic(nearby_agents_states)
        
        # Update Panic Level
        new_panic = current_state.panic + delta_time * (
            0.3 * hazard_intensity + 
            0.2 * social_panic + 
            0.1 * (1 - current_state.stamina) -
            self.emotional_decay_rate * current_state.panic
        )
        
        # Update stress level
        new_stress = current_state.stress + delta_time * (
            self.stress_accumulation_rate * new_panic +
            0.1 * hazard_intensity -
            self.emotional_decay_rate * current_state.stress
        )
        
        # Update stamina
        new_stamina = current_state.stamina - delta_time * (
            0.1 * new_panic +
            0.05 * new_stress
        )
        
        new_panic = np.clip(new_panic, 0.0, 1.0)
        new_stress = np.clip(new_stress, 0.0, 1.0)
        new_stamina = np.clip(new_stamina, 0.0, 1.0)
        
        return EmotionalState(
            panic=new_panic,
            stress=new_stress,
            stamina=new_stamina
        )
        
    def __calculate_hazard_influence(self, nearby_hazards: torch.Tensor) -> float:
        """Calculate influence of nearby hazards."""
        if len(nearby_hazards) == 0:
            return 0.0
        
        # Calculate distance-weighted hazard intensity
        distances = torch.norm(nearby_hazards[:, :2], dim=1)
        intensities = nearby_hazards[:, 2]
        
        # Apply inverse square law for hazard influence
        influence = (intensities / (1 + distances ** 2)).sum().item()
        return min(1.0, influence)
    
    def __calculate_social_panic(self, nearby_states: List[EmotionalState]) -> float:
        """Calculate panic influence from nearby agents."""
        if not nearby_states:
            return 0.0
        
        # Calculate average panic of nearby agents
        total_panic = sum(state.panic for state in nearby_states)
        avg_panic = total_panic / len(nearby_states)
        
        # Apply nonlinear social influence
        social_factor = 1.0 - np.exp(-2.0 * avg_panic)
        return social_factor