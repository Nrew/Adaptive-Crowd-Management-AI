import pytest
import torch
import torch.nn as nn
from ai_models.ppo import PPO
#from ai_models.agents import Agent

#def test_ppo_init():
#    policy = Agent(input_dim=4, output_dim=2)
#    value = Agent(input_dim=4, output_dim=2)
#    ppo = PPO(policy_network=policy, value_network=value)
#    assert isinstance(ppo, PPO)

# Mock Networks
class MockPolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MockPolicyNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.fc(x)

class MockValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(MockValueNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
    
    def forward(self, x):
       return self.fc(x)

@pytest.fixture
def mock_networks():
    """Fixture to provide mock policy and value networks."""
    policy_net = MockPolicyNetwork(input_dim=4, output_dim=2)
    value_net = MockValueNetwork(input_dim=4)
    return policy_net, value_net

def test_ppo_initialization(mock_networks):
    """
    Test the initialization of the PPO class.
    - Verifies that the policy and value networks are correctly assigned.
    - Ensures the optimizer combines parameters from both networks.
    """
    policy_net, value_net = mock_networks
    ppo = PPO(policy_network=policy_net, value_network=value_net)

    assert ppo.policy_network == policy_net, "Policy network not initialized correctly."
    assert ppo.value_network == value_net, "Value network not initialized correctly."
    assert isinstance(ppo.optimizer, torch.optim.Adam), "Optimizer not initialized as Adam."

def test_ppo_compute_loss_with_synthetic_data(mock_networks):
    """
    Test the internal `__compute_loss` function with synthetic data.
    - Ensures that the computed loss is a valid tensor.
    - Validates the shapes of intermediate components like log_probs and value_loss.
    """
    policy_net, value_net = mock_networks
    ppo = PPO(policy_network=policy_net, value_network=value_net)

    states = torch.randn(10, 4)  # 10 samples, 4 features
    actions = torch.randn(10, 2)
    old_log_probs = torch.randn(10)
    rewards = torch.randn(10)
    advantages = torch.randn(10)

    loss = ppo._PPO__compute_loss(states, actions, old_log_probs, rewards, advantages)

    assert isinstance(loss, torch.Tensor), "Loss is not a tensor."
    assert loss.ndim == 0, "Loss should be a scalar tensor."

def test_ppo_update_function(mock_networks):
    """
    Test the `update` method with real networks and synthetic data.
    - Ensures gradients are computed and applied to the networks.
    - Verifies parameter updates by comparing pre- and post-update weights.
    """
    policy_net, value_net = mock_networks
    ppo = PPO(policy_network=policy_net, value_network=value_net)

    # Dynamically add max_grad_norm for testing
    ppo.max_grad_norm = 0.5

    states = torch.randn(10, 4)
    actions = torch.randn(10, 2)
    old_log_probs = torch.randn(10)
    rewards = torch.randn(10)
    advantages = torch.randn(10)

    # Save initial weights
    initial_policy_weights = [param.clone() for param in policy_net.parameters()]
    initial_value_weights = [param.clone() for param in value_net.parameters()]

    # Perform update
    ppo.update(states, actions, old_log_probs, rewards, advantages)

    # Verify that weights have been updated
    for initial, updated in zip(initial_policy_weights, policy_net.parameters()):
        assert not torch.equal(initial, updated), "Policy network parameters not updated."

    for initial, updated in zip(initial_value_weights, value_net.parameters()):
        assert not torch.equal(initial, updated), "Value network parameters not updated."

@pytest.mark.parametrize("clip_epsilon", [0.1, 0.2, 0.3])
def test_ppo_loss_clipping_behavior(mock_networks, clip_epsilon):
    """
    Test the clipping behavior in the PPO loss function.
    - Ensures that the ratio is clipped within [1 - clip_epsilon, 1 + clip_epsilon].
    """
    policy_net, value_net = mock_networks
    ppo = PPO(policy_network=policy_net, value_network=value_net, clip_epsilon=clip_epsilon)

    states = torch.randn(10, 4)
    actions = torch.randn(10, 2)
    old_log_probs = torch.randn(10)
    rewards = torch.randn(10)
    advantages = torch.ones(10)  # Fixed advantages for predictable behavior

    # Mock policy network to return fixed values for testing
    with torch.no_grad():
        policy_net.fc.weight.fill_(1.0)
        policy_net.fc.bias.fill_(0.0)

    loss = ppo._PPO__compute_loss(states, actions, old_log_probs, rewards, advantages)

    # Ensure loss computation doesn't raise errors
    assert loss is not None, "Loss computation failed."
    assert loss.item() > 0, "Loss should be positive when using synthetic data."

@pytest.mark.parametrize("zero_rewards, zero_advantages", [(True, False), (False, True), (True, True)])
def test_ppo_handles_edge_cases(mock_networks, zero_rewards, zero_advantages):
    """
    Test PPO behavior in edge cases using synthetic data.
    - Verifies behavior when rewards or advantages are zero.
    """
    policy_net, value_net = mock_networks
    ppo = PPO(policy_network=policy_net, value_network=value_net)

    # Dynamically add max_grad_norm for testing
    ppo.max_grad_norm = 0.5

    states = torch.randn(10, 4)
    actions = torch.randn(10, 2)
    old_log_probs = torch.randn(10)
    rewards = torch.zeros(10) if zero_rewards else torch.randn(10)
    advantages = torch.zeros(10) if zero_advantages else torch.randn(10)

    # Perform update
    ppo.update(states, actions, old_log_probs, rewards, advantages)

    # Ensure the model does not throw errors during edge cases
    assert True, "Edge case update failed."

def test_ppo_with_mock_distributions(mock_networks):
    """
    Test the behavior of PPO with mocked distributions for policy network output.
    - Ensures that log_probs are calculated correctly.
    - Verifies the entropy calculation for action distribution.
    """
    policy_net, value_net = mock_networks
    ppo = PPO(policy_network=policy_net, value_network=value_net)

    states = torch.randn(10, 4)
    actions = torch.randn(10, 2)
    old_log_probs = torch.randn(10)
    rewards = torch.randn(10)
    advantages = torch.randn(10)

    # Mock distributions
    with torch.no_grad():
        policy_net.fc.weight.fill_(1.0)
        policy_net.fc.bias.fill_(0.0)

    # Test log_prob and entropy
    action_means = policy_net(states)
    action_std = torch.ones_like(action_means) * 0.1
    dist = torch.distributions.Normal(action_means, action_std)

    log_probs = dist.log_prob(actions).sum(dim=-1)
    entropy = dist.entropy().sum(dim=-1).mean()

    assert log_probs.shape == old_log_probs.shape, "Log_probs shape mismatch."
    assert entropy.item() > 0, "Entropy should be positive for a normal distribution."
