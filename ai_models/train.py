import os
import sys
import signal
import yaml
import logging
from typing import List

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agents import PPOAgent
from agents.ENNAgent import PPOAgentWithENN
from environment.unity_wrapper import UnityEnvironmentWrapper
from ppo import PPO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    print(config_path)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float,
    lam: float
) -> torch.Tensor:
    """Compute Generalized Advantage Estimation (GAE)."""
    advantages = []
    gae = 0
    next_value = 0
    for step in reversed(range(len(rewards))):
        mask = 1.0 - float(dones[step])
        delta = rewards[step] + gamma * next_value * mask - values[step]
        gae = delta + gamma * lam * gae * mask
        advantages.insert(0, gae)
        next_value = values[step]
    return torch.tensor(advantages, dtype=torch.float32)


def save_model(model: torch.nn.Module, input_dim: int, model_name: str) -> None:
    """Save the model in ONNX format."""
    os.makedirs('trained-models', exist_ok=True)
    dummy_input = torch.randn(1, input_dim)
    torch.onnx.export(
        model,
        dummy_input,
        f"trained-models/{model_name}.onnx",
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                      'output': {0: 'batch_size'}}
    )

class Trainer:
    """Trainer class to manage the training process."""

    def __init__(self, config: dict):
        self.config = config
        self.env = self._initialize_environment()
        self.agent = self._initialize_agent()
        self.ppo = self._initialize_ppo()
        self.writer = SummaryWriter()
        self._register_signal_handlers()
        self.global_step = 0

    def _initialize_environment(self) -> UnityEnvironmentWrapper:
        env_config = self.config['environment']
        env = UnityEnvironmentWrapper(
            file_name=env_config.get('file_name'),
            worker_id=env_config.get('worker_id', 0),
            base_port=env_config.get('base_port', 5004),
            no_graphics=env_config.get('no_graphics', False),
            seed=env_config.get('seed', 12345),
            time_scale=env_config.get('time_scale', 1.0),
            width=env_config.get('width', 720),
            height=env_config.get('height', 480),
            quality_level=env_config.get('quality_level', 0),
            target_frame_rate=env_config.get('target_frame_rate', 60)
        )
        return env

    def _initialize_agent(self):
        observation_dim = self.env.get_observation_spec().shape[0]
        action_dim = self.env.get_action_spec().continuous_size
        hidden_size = self.config['training'].get('hidden_size', 128)
        use_enn = self.config['training'].get('use_enn', False)

        if use_enn:
            enn_config = self.config['enn_config']
            agent = PPOAgentWithENN(
                observation_dim,
                action_dim,
                hidden_size=hidden_size,
                enn_config=enn_config
            )
        else:
            agent = PPOAgent(
                observation_dim,
                action_dim,
                hidden_size=hidden_size
            )

        self.use_enn = use_enn
        self.observation_dim = observation_dim
        return agent

    def _initialize_ppo(self):
        training_config = self.config['training']
        learning_rate = training_config.get('learning_rate', 3e-4)
        clip_param = training_config.get('clip_param', 0.2)
        max_grad_norm = training_config.get('max_grad_norm', 0.5)
        entropy_coef = training_config.get('entropy_coef', 0.01)

        ppo = PPO(
            policy_network=self.agent.policy_net,
            value_network=self.agent.value_net,
            clip_param=clip_param,
            lr=learning_rate,
            max_grad_norm=max_grad_norm,
            entropy_coef=entropy_coef
        )
        return ppo

    def _register_signal_handlers(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        logger.info('\nSaving model before exit...')
        self.save()
        self.env.close()
        sys.exit(0)

    def train(self):
        training_config = self.config['training']
        num_episodes = training_config.get('num_episodes', 1000)
        max_steps_per_episode = training_config.get('max_steps_per_episode', 1000)
        gamma = training_config.get('gamma', 0.99)
        lam = training_config.get('lam', 0.95)

        for episode in range(num_episodes):
            self.env.reset()
            episode_reward = 0
            steps = 0
            done = False

            states, actions, rewards, log_probs, values, dones = []

            logger.info(f"Starting Episode {episode}")

            while not done and steps < max_steps_per_episode:
                obs_batch, reward_batch, done_batch = self.env.get_obs()

                if len(obs_batch) == 0:
                    logger.warning("No agents found in environment")
                    break

                actions_batch = []
                log_probs_batch = []
                values_batch = []

                for idx, obs in enumerate(obs_batch):
                    state_tensor = torch.tensor(obs, dtype=torch.float32)

                    if self.use_enn:
                        # Implement retrieval of nearby agents and hazards
                        nearby_agents = None
                        nearby_hazards = None
                        action, log_prob, value = self.agent.act(
                            state_tensor, nearby_agents, nearby_hazards
                        )
                    else:
                        action, log_prob, value = self.agent.act(state_tensor)

                    actions_batch.append(action.detach().numpy())
                    log_probs_batch.append(log_prob.item())
                    values_batch.append(value.item())

                    # Store experiences
                    states.append(state_tensor)
                    actions.append(torch.tensor(action.detach(), dtype=torch.float32))
                    rewards.append(reward_batch[idx])
                    log_probs.append(log_prob.item())
                    values.append(value.item())
                    dones.append(done_batch[idx])

                actions_array = np.vstack(actions_batch)
                self.env.set_actions(actions_array)
                self.env.step()

                episode_reward += sum(reward_batch)
                steps += 1
                self.global_step += 1

                if any(done_batch):
                    done = True

            if len(rewards) > 0:
                advantages = compute_gae(rewards, values, dones, gamma, lam)
                returns = advantages + torch.tensor(values, dtype=torch.float32)

                # Convert lists to tensors
                states_tensor = torch.stack(states)
                actions_tensor = torch.stack(actions)
                old_log_probs_tensor = torch.tensor(log_probs, dtype=torch.float32)

                # Update the policy and value networks
                self.ppo.update(
                    states_tensor,
                    actions_tensor,
                    old_log_probs_tensor,
                    returns,
                    advantages
                )

                # Logging
                self.writer.add_scalar('Episode Reward', episode_reward, episode)
                logger.info(
                    f"Episode {episode} completed. "
                    f"Total Reward: {episode_reward:.2f}, "
                    f"Steps: {steps}"
                )

        self.save()
        self.env.close()

    def save(self):
        model_name = 'policy_network_with_enn' if self.use_enn else 'policy_network'
        save_model(self.agent, self.observation_dim, model_name)


if __name__ == "__main__":
    config_path = 'ai_models\config\config.yaml' 
    config = load_config(config_path)
    trainer = Trainer(config)
    trainer.train()
