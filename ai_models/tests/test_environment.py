import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from ai_models.environment.unity_wrapper import UnityEnvironmentWrapper

# TODO: Test enviroment when its all set up.
#def test_enviroment_init():
#    env = UnityEnvWrapper(file_name="")
#    assert env.behavior_name is not None
#    env.close()

# Mocking our UnityEnvironment along with related components. 

@pytest.fixture
def mock_unity_environment():
    """
    Fixture to mock the UnityEnvironment.
    - Replaces the UnityEnvironment class with a mock object.
    - Mocks behavior specifications, reset, step, and other methods.
    """
    with patch("ai_models.environment.unity_wrapper.UnityEnvironment") as mock_env_class:
        mock_env = MagicMock()
        mock_env.behavior_specs = {"TestBehavior": MagicMock()}
        mock_env.get_steps.return_value = (MagicMock(), MagicMock())
        mock_env_class.return_value = mock_env
        yield mock_env

# Test UnityEnvWrapper initialization
def test_environment_init(mock_unity_environment):
    """
    Test the initialization of UnityEnvironmentWrapper.
    - Ensures that the behavior name is correctly set from the mocked UnityEnvironment.
    - Verifies that the number of agents matches the length of agent IDs.
    - Ensures the environment's close method is called once the wrapper is closed.
    """
    wrapper = UnityEnvironmentWrapper(file_name="test_env")
    assert wrapper.behavior_name == "TestBehavior", "Behavior name should match the mocked behavior"
    assert wrapper.num_agents == len(wrapper.agent_ids), "Number of agents should match agent IDs"
    wrapper.close()
    mock_unity_environment.close.assert_called_once()

# Test reset functionality
def test_environment_reset(mock_unity_environment):
    """
    Test the reset functionality of UnityEnvironmentWrapper.
    - Ensures that the UnityEnvironment's reset method is called.
    - Verifies that resetting updates the agent count and IDs in the wrapper.
    """
    wrapper = UnityEnvironmentWrapper(file_name="test_env")
    wrapper.reset()
    # Assert reset has been called twice (once during init and once explicitly)
    assert mock_unity_environment.reset.call_count == 2, "Reset should be called twice: once in init, once explicitly."

# Test getting observations
def test_environment_get_obs(mock_unity_environment):
    """
    Test the get_obs method of UnityEnvironmentWrapper.
    - Mocks decision and terminal steps to simulate the environment's response.
    - Ensures the method returns observations, rewards, and dones correctly.
    - Verifies the types and validity of the returned values.
    """
    # Mock decision and terminal steps
    mock_decision_steps = MagicMock()
    mock_decision_steps.obs = [MagicMock()]
    mock_decision_steps.reward = [1.0]
    mock_terminal_steps = MagicMock()
    mock_terminal_steps.obs = [MagicMock()]
    mock_terminal_steps.reward = [-1.0]
    mock_unity_environment.get_steps.return_value = (mock_decision_steps, mock_terminal_steps)

    wrapper = UnityEnvironmentWrapper(file_name="test_env")
    obs, rewards, dones = wrapper.get_obs()
    assert obs is not None, "Observations should not be None"
    assert rewards is not None, "Rewards should not be None"
    assert isinstance(dones, np.ndarray), "Dones should be a numpy array"
    wrapper.close()

# Test setting actions
def test_environment_set_actions(mock_unity_environment):
    """
    Test the set_actions method of UnityEnvironmentWrapper.
    - Simulates sending actions to the UnityEnvironment.
    - Verifies that the set_actions method of UnityEnvironment is called correctly.
    """
    wrapper = UnityEnvironmentWrapper(file_name="test_env")
    actions = np.array([[0.1, 0.2]])
    wrapper.set_actions(actions)
    mock_unity_environment.set_actions.assert_called_once()
    wrapper.close()

# Test stepping the environment
def test_environment_step(mock_unity_environment):
    """
    Test the step method of UnityEnvironmentWrapper.
    - Simulates advancing the UnityEnvironment by one step.
    - Verifies that the UnityEnvironment's step method is called once.
    """
    wrapper = UnityEnvironmentWrapper(file_name="test_env")
    wrapper.step()
    mock_unity_environment.step.assert_called_once()
    wrapper.close()

# Test getting action specifications
def test_environment_get_action_spec(mock_unity_environment):
    """
    Test the get_action_spec method of UnityEnvironmentWrapper.
    - Mocks the action specification in the UnityEnvironment.
    - Ensures the wrapper correctly retrieves the mocked action spec.
    """
    mock_action_spec = MagicMock()
    mock_unity_environment.behavior_specs["TestBehavior"].action_spec = mock_action_spec

    wrapper = UnityEnvironmentWrapper(file_name="test_env")
    action_spec = wrapper.get_action_spec()
    assert action_spec == mock_action_spec, "Action spec should match the mocked spec"
    wrapper.close()

# Test getting observation specifications
def test_environment_get_observation_spec(mock_unity_environment):
    """
    Test the get_observation_spec method of UnityEnvironmentWrapper.
    - Mocks the observation specification in the UnityEnvironment.
    - Ensures the wrapper correctly retrieves the mocked observation spec.
    """
    mock_observation_spec = MagicMock()
    mock_unity_environment.behavior_specs["TestBehavior"].observation_specs = [mock_observation_spec]

    wrapper = UnityEnvironmentWrapper(file_name="test_env")
    observation_spec = wrapper.get_observation_spec()
    assert observation_spec == mock_observation_spec, "Observation spec should match the mocked spec"
    wrapper.close()

# Test getting agent IDs
def test_environment_get_agent_ids(mock_unity_environment):
    """
    Test the get_agent_ids method of UnityEnvironmentWrapper.
    - Mocks the agent IDs in the decision steps of the UnityEnvironment.
    - Ensures the wrapper correctly retrieves the mocked agent IDs.
    """
    mock_decision_steps = MagicMock()
    mock_decision_steps.agent_id = [1, 2, 3]
    mock_unity_environment.get_steps.return_value = (mock_decision_steps, MagicMock())

    wrapper = UnityEnvironmentWrapper(file_name="test_env")
    agent_ids = wrapper.get_agent_ids()
    assert agent_ids == [1, 2, 3], "Agent IDs should match the mocked IDs"
    wrapper.close()
