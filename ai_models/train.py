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
        timeout_wait=15,
        seed=12345
    )
    print("Successfully connected to Unity environment")

    channel.set_configuration_parameters(
        time_scale=5.0,
        width=720,
        height=480,
        quality_level=0,
        target_frame_rate=60
    )

    env.reset()

    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    print(f"Action spec:\n\tContinuous: {spec.action_spec.continuous_size}\n\tDiscrete: {spec.action_spec.discrete_branches}")
    print(f"\nFull Action spec: {spec.action_spec}")

    input_dim = spec.observation_specs[0].shape[0] # 8 input dimensions
    print(f"Input dimensions: {spec.observation_specs[0].shape[0]} ")
    output_dim = 2  #spec.action_spec.continuous_size
    
    policy_network = Agent(input_dim, output_dim) # 8 input dim, 2 output dim
    value_network = Agent(input_dim, 1)
    ppo = PPO(policy_network, value_network)

    
    try:
        for episode in range(1000):
            env.reset()
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            print(f"Decision steps: {decision_steps} \n\n\t Obs: {decision_steps.obs} \n\n\t Obs[0][0] = {decision_steps.obs[0][0]}")
            print(f"Obs shape: {decision_steps.obs[0][0].shape}")

            if len(decision_steps) == 0:
                print("No agents found in environment")
                continue

            states, actions, rewards, log_probs, values = [],[],[],[],[]
            done = False
            steps = 0
            print(f"Training episode {episode}")
            while not done and steps < 1000:
                state = decision_steps.obs[0][0]

                # get action from policy
                state_tensor = torch.tensor(state, dtype=torch.float32)
                action, log_prob = policy_network.act(state_tensor)

                # take action
                action_array = action.detach().reshape(1,2)
                continuous_action = np.zeros((1,2))
                continuous_action[0] = action_array
                discrete_actions = np.zeros((1,1), dtype=np.float32)
                discrete_actions[0,0] = 0
                action_tuple = ActionTuple(continuous=continuous_action, discrete=discrete_actions)
                env.set_actions(behavior_name, action_tuple)

                # next state and reward
                env.step()
                decision_steps, terminal_steps = env.get_steps(behavior_name)

                done = len(terminal_steps) > 0

                if done:
                    reward = terminal_steps.reward[0]
                else:
                    reward = decision_steps.reward[0] if len(decision_steps) > 0 else 0.0
                
                states.append(torch.from_numpy(state).float())
                actions.append(action.detach().clone())
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value_network(state_tensor).item())
                
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