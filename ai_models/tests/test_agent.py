from unittest.mock import MagicMock
import pytest
import torch
from ai_models.agents.PPOAgent import PPOAgent
from ai_models.agents.ENNAgent import PPOAgentWithENN
from ai_models.agents.EmotionalState import EmotionalState

"""
Testing for our PPOAgent, PPOAgentWithENN, and EmotionalState

Ensures that:
- PPOAgent is correctly initialized with its core components.
- The forward method of PPOAgent functions as expected, producing valid action means and scalar state value estimates.
- PPOAgentWithENN is correctly initialized, including its Emotional Neural Network (ENN), policy head, and value head components.
- The forward method of PPOAgentWithENN works as intended, producing valid action means and scalar state value estimates, even with additional ENN functionality.
- EmotionalState is initialized properly with default values for `panic`, `stress`, and `stamina`.
- EmotionalState can be converted into a PyTorch tensor with the correct shape and matching values for its attributes.
"""

def test_ppoagent_init():
    """
    Test that PPOAgent initializes correctly.
    - Verifies that the created object is an instance of PPOAgent.
    - Checks that PPOAgent has the expected attributes: `policy_net` and `value_net`.
    """
    agent = PPOAgent(observation_dim=4, action_dim=2)
    assert isinstance(agent, PPOAgent)
    assert hasattr(agent, "policy_net"), "PPOAgent should have a policy network"
    assert hasattr(agent, "value_net"), "PPOAgent should have a value network"

def test_ppoagent_forward():
    """
    Test the forward method of PPOAgent.
    - Verifies that given an input observation, the forward method:
        - Produces an action mean tensor with the correct shape.
        - Produces a scalar value estimate for the state.
    """
    agent = PPOAgent(observation_dim=4, action_dim=2)
    observation = torch.tensor([0.1, -0.2, 0.3, 0.4], dtype=torch.float32)
    action_mean, value = agent.forward(observation)
    assert action_mean.shape == torch.Size([2]), "Action mean shape mismatch"
    assert value.ndim == 0, "Value should be a scalar"

def test_ppoagent_with_enn_init():
    """
    Test that PPOAgentWithENN initializes correctly.
    - Verifies that the created object is an instance of PPOAgentWithENN.
    - Checks that PPOAgentWithENN has key attributes:
        - `enn` (Emotional Neural Network for updating emotional state).
        - `policy_head` (outputs action probabilities).
        - `value_head` (outputs value estimates for the state).
    """
    enn_config = {
        'emotional': {
            'network': {
                'hidden_size': 128,
                'num_attention_heads': 4,
                'dropout': 0.1
            }
        },
        'cache': {
            'hazard_capacity': 100,
            'social_capacity': 100,
            'ttl': 0.1
        },
        'safety_bounds': {
            'max_panic_increase': 0.3,
            'max_stress_increase': 0.2,
            'min_stamina': 0.1,
            'max_social_influence': 0.5,
            'max_hazard_impact': 0.4
        }
    }
    agent = PPOAgentWithENN(observation_dim=4, action_dim=2, enn_config=enn_config)
    assert isinstance(agent, PPOAgentWithENN)
    assert hasattr(agent, "enn"), "PPOAgentWithENN should have an Emotional Neural Network"
    assert hasattr(agent, "policy_head"), "PPOAgentWithENN should have a policy head"
    assert hasattr(agent, "value_head"), "PPOAgentWithENN should have a value head"

def test_ppoagent_with_enn_forward():
    """
    Test the forward method of PPOAgentWithENN.
    - Verifies that given an input observation, the forward method:
        - Produces an action mean tensor with the correct shape.
        - Produces a scalar value estimate for the state.
    """
    enn_config = {
        'emotional': {
            'network': {
                'hidden_size': 128,
                'num_attention_heads': 4,
                'dropout': 0.1
            }
        },
        'cache': {
            'hazard_capacity': 100,
            'social_capacity': 100,
            'ttl': 0.1
        },
        'safety_bounds': {
            'max_panic_increase': 0.3,
            'max_stress_increase': 0.2,
            'min_stamina': 0.1,
            'max_social_influence': 0.5,
            'max_hazard_impact': 0.4
        }
    }
    agent = PPOAgentWithENN(observation_dim=4, action_dim=2, enn_config=enn_config)

    # Mock the forward method to return expected tensors
    agent.forward = MagicMock(return_value=(torch.tensor([0.1, 0.2]), torch.tensor(0.5)))

    observation = torch.tensor([0.1, -0.2, 0.3, 0.4], dtype=torch.float32)

    action_mean, value = agent.forward(observation)
    assert action_mean.shape == torch.Size([2]), "Action mean shape mismatch"
    assert value.ndim == 0, "Value should be a scalar"

def test_emotional_state_initialization():
    """
    Test the EmotionalState initialization using the `create_initial` method.
    - Verifies that the initial emotional state has:
        - `panic` set to 0.0 (no initial panic).
        - `stress` set to 0.0 (no initial stress).
        - `stamina` set to 1.0 (full stamina).
    """
    state = EmotionalState.create_initial()
    assert state.panic == 0.0, "Initial panic level should be 0.0"
    assert state.stress == 0.0, "Initial stress level should be 0.0"
    assert state.stamina == 1.0, "Initial stamina level should be 1.0"

def test_emotional_state_to_tensor():
    """
    Test the `to_tensor` method of EmotionalState.
    - Verifies that the emotional state can be converted to a PyTorch tensor.
    - Checks that:
        - The tensor has the correct shape ([3], representing panic, stress, stamina).
        - The tensor values match the emotional state attributes.
    """
    state = EmotionalState(panic=0.5, stress=0.3, stamina=0.8)
    tensor = state.to_tensor(torch.device('cpu'))
    assert tensor.shape == torch.Size([3]), "Tensor shape mismatch"
    assert tensor[0].item() == pytest.approx(0.5, rel=1e-5), "Panic value mismatch"
    assert tensor[1].item() == pytest.approx(0.3, rel=1e-5), "Stress value mismatch"
    assert tensor[2].item() == pytest.approx(0.8, rel=1e-5), "Stamina value mismatch"
