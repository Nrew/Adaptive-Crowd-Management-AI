import torch 
from ppo import PPO
from agent import Agent
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from typing import List
from mlagents_envs.base_env import ActionTuple
import numpy as np

def compute_advantages(rewards: List[float], values: List[float], gamma: float = 0.99) -> torch.Tensor:
    advantages = []
    discounted_sum = 0
    for reward, value in zip(reversed(rewards), reversed(values)):
        discounted_sum = reward + gamma * discounted_sum
        advantages.insert(0, discounted_sum - value)
    return torch.tensor(advantages, dtype=torch.float32)

def main():
    print("Starting training script...")
    print("Attempting to connect to Unity environment...")
    channel = EngineConfigurationChannel()
    print("Channel connected...")
    env = UnityEnvironment(
        file_name=None,
        side_channels=[channel],
        worker_id=0,
        base_port=5004,
        no_graphics=False,
        timeout_wait=60,
        seed=12345
    )
    print("Successfully connected to Unity environment")

    channel.set_configuration_parameters(
        time_scale=10.0,
        width=720,
        height=480,
        quality_level=0,
        target_frame_rate=60
    )

    env.reset()

    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    print(f"Action spec:\n\tContinuous: {spec.action_spec.continuous_size} \n\tDiscrete: {spec.action_spec.discrete_branches}")
    print(f"\nFull Action spec: {spec.action_spec}")

    input_dim = spec.observation_specs[0].shape[0] # 9 input dimensions
    print(f"Input dimensions: {spec.observation_specs[0].shape[0]} ")
    output_dim = 2  #spec.action_spec.continuous_size
    
    policy_network = Agent(input_dim, output_dim) # 9 input dim, 2 output dim
    value_network = Agent(input_dim, 1)
    ppo = PPO(policy_network, value_network)

    
    try:
        for episode in range(1001):
            env.reset()
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            # print(f"Decision steps: {decision_steps} \n\n\t Obs: {decision_steps.obs} \n\n\t Obs[0][0] = {decision_steps.obs[0][0]}")
            # print(f"Obs shape: {decision_steps.obs[0][0].shape}")

            if len(decision_steps) == 0:
                print("No agents found in environment")
                continue

            states, actions, rewards, log_probs, values = [],[],[],[],[]
            done = False
            steps = 0


            print(f"Training episode {episode}")
            while not done and steps <= 1000:
                if steps % 100 == 0:
                    print(f"Step: {steps}")
                # state = decision_steps.obs[0][0]
                state = decision_steps.obs[0]

                current_actions = []
                log_probs_batch = []

                for agent_index in range(len(state)):
                    state_tensor = torch.tensor(state[agent_index], dtype=torch.float32)
                    action, log_prob = policy_network.act(state_tensor)
                    current_actions.append(action)
                    log_probs_batch.append(log_prob)

                # # get action from policy for each individual agent
                action_array = np.stack([a.detach().numpy() for a in current_actions])
                continuous_actions = action_array.reshape(len(current_actions), 2)

                # take action
                action_tuple = ActionTuple(continuous=continuous_actions, discrete=None)
                env.set_actions(behavior_name, action_tuple)

                # next state and reward
                env.step()
                decision_steps, terminal_steps = env.get_steps(behavior_name)

                done = len(terminal_steps) > 0

                if done:
                    reward = terminal_steps.reward
                else:
                    reward = decision_steps.reward if len(decision_steps) > 0 else np.zeros(len(actions))
                for agent_index in range(len(state)):
                    states.append(torch.from_numpy(state[agent_index]).float())
                    actions.append(current_actions[agent_index].detach().clone())
                    rewards.append(reward[agent_index])
                    log_probs.append(log_probs_batch[agent_index])
                    values.append(value_network(torch.tensor(state[agent_index])).item())
                steps += 1
                
            if len(states) > 0:
                advantages = compute_advantages(rewards, values)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
                states_tensor = torch.stack(states)
                actions_tensor = torch.stack(actions)
                log_probs_tensor = torch.tensor([lp.item() for lp in log_probs], dtype=torch.float32)
                ppo.update(states_tensor, actions_tensor, log_probs_tensor, rewards_tensor, advantages)
                print(f"Episode {episode} completed with {len(states)} steps and final reward {sum(rewards):.2f}")

    except KeyboardInterrupt:
        print(f"Training interrupted by user")
    finally:  
        env.close()
    
if __name__ == "__main__":
    main()