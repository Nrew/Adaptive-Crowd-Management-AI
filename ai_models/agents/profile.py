from dataclasses import dataclass
import numpy as np
import torch

@dataclass(frozen=True)
class AgentProfile:
    """
    Immutable agent personaility traits configuration.
    """
    base_panic_threshold: float # Base threshold for panic behvaior
    social_influence: float     # How much an agent is influenced by others (0-1)
    helping_tendency: float     # Probablity of helping others
    decision_speed: float       # How quickly agent makes decisions
    risk_aversion: float        # Tendency to avoid dangerous situations
    emotional_stability: float  # Resistance to panic (0-1)
    leadership_score: float     # Influence on other agents (0-1)

    def to_tensor(self, device: torch.device) -> torch.Tensor:
        """
        Convert profile to tensor

        Args:
            device (torch.device): CPU/GPU

        Returns:
            torch.Tensor: Pytorch tensor representing the agent profile
        """
        return torch.tensor([
            self.base_panic_threshold,
            self.social_influence,
            self.helping_tendency,
            self.decision_speed,
            self.risk_aversion,
            self.emotional_stability,
            self.leadership_score
        ], dtype=torch.float32, device=device)

class ProfileFactory:
    """Factory for creating agent profiles."""
    
    @staticmethod
    def create_random_profile() -> AgentProfile:
        """Create a random agent profile with realistic distributions."""
        return AgentProfile(
            base_panic_threshold=np.random.normal(0.7, 0.1),
            social_influence=np.random.beta(2, 2),
            helping_tendency=np.random.beta(2, 3),
            decision_speed=np.random.normal(0.5, 0.1),
            risk_aversion=np.random.beta(2, 2),
            emotional_stability=np.random.beta(3, 2),
            leadership_score=np.random.beta(1, 4)
        )
    
    @staticmethod
    def create_leader_profile() -> AgentProfile:
        """Create a profile for a leader agent."""
        return AgentProfile(
            base_panic_threshold=0.8,
            social_influence=0.9,
            helping_tendency=0.8,
            decision_speed=0.7,
            risk_aversion=0.4,
            emotional_stability=0.9,
            leadership_score=0.9
        )