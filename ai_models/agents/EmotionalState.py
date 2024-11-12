from typing import Protocol, NamedTuple
import torch
from dataclasses import dataclass


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