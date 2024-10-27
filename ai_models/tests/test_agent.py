import torch
from agent import Agent

# TODO: Make more comprehensive tests
def test_agent_init():
    agent = Agent(input_dim=4, output_dim=4)
    assert isinstance(agent, Agent)

def test_agent_act():
    agent = Agent(input_dim=4, output_dim=2)
    state = torch.tensor([0.1, -0.2, 0.3, 0.4], dtype=torch.float32)
    action, log_prob = agent.act(state)
    assert isinstance(action, int)
    assert isinstance(log_prob, torch.tensor)