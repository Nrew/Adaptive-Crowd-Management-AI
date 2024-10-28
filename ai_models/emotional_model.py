import torch
import torch.nn as nn
from typing import Dict

class EmotionalModel(nn.Module):
    def __init__(self, input_dim: int):
        super(EmotionalModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        self.network(state)
        
    def predict_panic_level(self, enviromental_factors: Dict[str, float]) -> float:
        state = torch.tensor([enviromental_factors[key] for key in sorted(enviromental_factors.keys())])
        panic_level = self.forward(state).item()
        return panic_level