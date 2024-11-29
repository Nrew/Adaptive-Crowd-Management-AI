import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim

class PPO:
    def __init__(self,
                 policy_network: nn.Module,
                 value_network: nn.Module,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 clip_epsilon: float = 0.2
    ) -> None:
        """
        Proximal Policy Optimization Algorithm.

        Args:
            policy_network (nn.Module): The policy network (actor).
            value_network (nn.Module): The value network (critic)
            lr (float, optional): Learning rate. Defaults to 3e-4.
            gamma (float, optional): Discount factor for rewards. Defaults to 0.99.
            clip_epsilon (float, optional): Clipping parameter for PPO. Defaults to 0.2.
        """
        self.policy_network = policy_network
        self.value_network = value_network
        self.optimizer = optim.Adam(
            list(self.policy_network.parameters()) + list(self.value_network.parameters()),
            lr=lr
        )
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        
    def __compute_loss(self,
                     states: torch.Tensor,
                     actions: torch.Tensor,
                     old_log_probs: torch.Tensor,
                     rewards: torch.Tensor,
                     adavntages: torch.Tensor) -> torch.Tensor:
        """
        Computes the PPO loss.

        Args:
            states (torch.Tensor): _description_
            actions (torch.Tensor): _description_
            old_log_probs (torch.Tensor): _description_
            rewards (torch.Tensor): _description_
            adavntages (torch.Tensor): _description_

        Returns:
            torch.Tensor: Combined policy and value loss.
        """
        action_means = self.policy_network(states)
        action_std = torch.ones_like(action_means) * 0.1
        dist = torch.distributions.Normal(action_means, action_std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()

        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        policy_loss = -torch.min(ratio * adavntages, clipped_ratio * adavntages).mean()
        
        value_estimates = self.value_network(states).squeeze(1)
        value_loss = F.mse_loss(value_estimates, rewards)
        
        return policy_loss + 0.5 * value_loss - (0.01 * entropy)
    
    def update(self,
                     states: torch.Tensor,
                     actions: torch.Tensor,
                     old_log_probs: torch.Tensor,
                     rewards: torch.Tensor,
                     advantages: torch.Tensor) -> None:
        """
        Update the policy and value networks based on PPO loss.

        Args:
            states (torch.Tensor): _description_
            actions (torch.Tensor): _description_
            old_log_probs (torch.Tensor): _description_
            rewards (torch.Tensor): _description_
            adavntages (torch.Tensor): _description_
        """
        loss = self.__compute_loss(
            states, actions, old_log_probs, rewards, advantages
        )

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy_network.parameters()) + list(self.value_network.parameters()),
            self.max_grad_norm
        )
        self.optimizer.step()