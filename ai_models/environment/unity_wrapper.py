import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple

class UnityEnvironmentWrapper:
    def __init__(
        self,
        file_name=None,
        worker_id=0,
        base_port=5004,
        no_graphics=False,
        seed=12345,
        time_scale=1.0,
        width=720,
        height=480,
        quality_level=0,
        target_frame_rate=60
    ) -> None:
        self.channel = EngineConfigurationChannel()
        self.env = UnityEnvironment(
            file_name=file_name,
            side_channels=[self.channel],
            worker_id=worker_id,
            base_port=base_port,
            no_graphics=no_graphics,
            seed=seed
        )
        self.env.reset()
        self.behavior_name = list(self.env.behavior_specs.keys())[0]
        self.spec = self.env.behavior_specs[self.behavior_name]
        self.channel.set_configuration_parameters(
            time_scale=time_scale,
            width=width,
            height=height,
            quality_level=quality_level,
            target_frame_rate=target_frame_rate
        )
        self.decision_steps, self.terminal_steps = self.env.get_steps(self.behavior_name)
        self.agent_ids = self.decision_steps.agent_id
        self.num_agents = len(self.agent_ids)
    
    def reset(self):
        self.env.reset()
        self.decision_steps, self.terminal_steps = self.env.get_steps(self.behavior_name)
        self.agent_ids = self.decision_steps.agent_id
        self.num_agents = len(self.agent_ids)
    
    def get_obs(self):
        if len(self.decision_steps) > 0:
            obs = self.decision_steps.obs[0]
            rewards = self.decision_steps.reward
            dones = np.zeros(len(obs), dtype=bool)
        else:
            obs = self.terminal_steps.obs[0]
            rewards = self.terminal_steps.reward
            dones = np.ones(len(obs), dtype=bool)
        return obs, rewards, dones
    
    def set_actions(self, actions):
        action_tuple = ActionTuple(continuous=actions)
        self.env.set_actions(self.behavior_name, action_tuple)
    
    def step(self):
        self.env.step()
        self.decision_steps, self.terminal_steps = self.env.get_steps(self.behavior_name)
    
    def close(self):
        self.env.close()
    
    def get_action_spec(self):
        return self.spec.action_spec
    
    def get_observation_spec(self):
        return self.spec.observation_specs[0]
    
    def get_agent_ids(self):
        return self.agent_ids
