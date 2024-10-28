from mlagents_envs.enviroment import UnityEnviroment
from mlagents_envs.base_env import ActionTuple
import numpy as np
from typing import Tuple

class UnityEnvWrapper:
    """
    Wrapper for Unity ML-Agents enviroment.
    """

    def __init__(self, file_name: str = None):
        self.env = UnityEnviroment(file_name=file_name)
        self.env.reset()
        self.behavior_name = list(self.env.behavior_specs.keys())[0]
        self.spec = self.env.behavior_specs[self.behavior_name]
        
    def get_state(self) -> np.ndarray:
        """
        Get the current state from the environment.

        Returns:
            np.ndarray: A Numpy representation of the current state.
        """
        decision_steps, _ = self.env.get_steps(self.behavior_name)
        return np.array(decision_steps.obs[0][0], dtype=np.float32)
    
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