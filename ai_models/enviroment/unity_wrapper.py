from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple, DecisionSteps, TerminalSteps
from typing import Tuple, List, Optional, Dict
import numpy as np
import torch
from dataclasses import dataclass

@dataclass
class EnvironmentState:
    """Represents the current state of the environment."""
    
    agent_states: torch.Tensor  # Shape: [num_agents, state_size]
    global_state: torch.Tensor  # Shape: [global_state_size]
    masks: torch.Tensor        # Shape: [num_agents]
    panic_levels: torch.Tensor  # Shape: [num_agents]

class UnityEnvWrapper:
    """
    Wrapper for Unity ML-Agents enviroment.

    This class handles:
    1. Environment initialization and reset
    2. Step execution and state management
    3. Observation and action processing
    4. Multi-agent coordination
    """
    def __init__(self,
                file_name: Optional[str] = None,
                worker_id = 0,
                time_scale: float = 1.0) -> None:
        """
        Initialize Unity environment wrapper.

        Args:
            file_name: Path to Unity executable
            worker_id: Unique identifier for this environment instance
            time_scale: Unity time scale factor
        """
        self.env = UnityEnvironment(
            file_name=file_name,
            worker_id=worker_id,
            time_scale=time_scale
        )
        
        self.env.reset()
        
        # Get behavior specs
        self.behavior_name = list(self.env.behavior_specs.keys())[0]
        self.spec = self.env.behavior_specs[self.behavior_name]
        
        # Cache Dimentions
        self.state_size = sum(shape[0] for shape in self.spec.observation_specs)
        self.action_size = self.spec.action_spec.continuous_size
        
        # Initialize buffers
        self.current_state: Optional[EnvironmentState] = None
        self.time_scale = time_scale
        
    def get_state(self) -> np.ndarray:
        """
        Get the current state from the environment.

        Returns:
            np.ndarray: A Numpy representation of the current state.
        """
        decision_steps, _ = self.env.get_steps(self.behavior_name)
        return np.array(decision_steps.obs[0][0], dtype=np.float32)
    def reset(self) -> EnvironmentState:
        """
        Reset enviroment to the initial state.

        Returns:
            EnvironmentState: _description_
        """
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        Send an action to the enviroment and get the result.

        Args:
            action (np.ndarray): _description_

        Returns:
            Tuple[np.ndarray, float, bool]: next_state, reward, and done flag.
        """
        action_tuple = ActionTuple(continious=np.array([action], dtype=np.float32))
        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()
        
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        if len(terminal_steps) > 0:
            reward = terminal_steps.reward[0]
            done = True
            next_state = terminal_steps.obs[0][0]
        else:
            reward = decision_steps.reward[0]
            done = False
            next_state = decision_steps.obs[0][0]
        
        return np.array(next_state, dtype=np.float32), reward, done
    
    def close(self):
        """
        Close the Unity Enviroment.
        """
        self.env.close()