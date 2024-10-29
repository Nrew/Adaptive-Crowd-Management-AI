import torch 
from ppo import PPO
from agent import Agent
from ai_models.enviroment.unity_wrapper import UnityEnviroment
from typing import List

def compute_advantages(rewards: List[float], values: List[float], gamma: float = 0.99) -> torch.Tensor:
    advantages = []
    discounted_sum = 0
    for reward, value in zip(reversed(rewards), reversed(values)):
        discounted_sum = reward + gamma * discounted_sum
        advantages.insert(0, discounted_sum - value)
    return torch.tensor(advantages, dtype=torch.float32)

def main():
    env = UnityEnviroment(file_name="")
    input_dim = env.spec.observation_shapes[0][0]
    output_dim = env.spec.action_spec.continious_size
    
    policy_network = Agent(input_dim, output_dim)
    value_network = Agent(input_dim, 1)
    ppo = PPO(policy_network, value_network)
    
    for episode in range(1000):
        states, actions, rewards, log_probs, values = [],[],[],[],[]
        state = env.get_state()
        
        done = False
        while not done:
            action, log_prob = policy_network.act(torch.tensor(state, dtype=torch.float32))
            next_state, reward, done = env([action])
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value_network(torch.tensor(state, dtype=torch.float32)))
            
            state = next_state
        advantages = compute_advantages(rewards, values)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.int64)
        log_probs_tensor = torch.tensor([lp.item() for lp in log_probs], dtype=torch.float32)
        
        ppo.update(states_tensor, actions_tensor, log_probs_tensor, rewards_tensor, advantages)
        
    env.close()
    
if __name__ == "__main__":
    main()