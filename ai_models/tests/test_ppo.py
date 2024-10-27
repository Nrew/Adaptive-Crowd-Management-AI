import torch
from ppo import PPO
from agent import Agent

def test_ppo_init():
    policy = Agent(input_dim=4, output_dim=2)
    value = Agent(input_dim=4, output_dim=2)
    ppo = PPO(policy_network=policy, value_network=value)
    assert isinstance(ppo, PPO)