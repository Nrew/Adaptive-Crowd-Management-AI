import torch
import torch as nn
import torch.optim as optim

class PPO:
    def __init__(self,
                 policy_network: nn.Module,
                 value_network: nn.Module,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 clip_epsilon: float = 0.2) -> None:
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
                     log_prob_old: torch.Tensor,
                     rewards: torch.Tensor,
                     adavntages: torch.Tensor) -> torch.Tensor:
        """
        Computes the PPO loss.

        Args:
            states (torch.Tensor): _description_
            actions (torch.Tensor): _description_
            log_prob_old (torch.Tensor): _description_
            rewards (torch.Tensor): _description_
            adavntages (torch.Tensor): _description_

        Returns:
            torch.Tensor: Combined policy and value loss.
        """
        log_prob_new = self.policy_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        ratio = torch.exp(log_prob_new - log_prob_old)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        policy_loss = -torch.min(ratio * adavntages, clipped_ratio * adavntages).mean()
        
        value_estimates = self.value_network(states).squeeze(1)
        value_loss = nn.MSELoss()(value_estimates, rewards)
        
        return policy_loss + 0.5 * value_loss
    def update(self,
                     states: torch.Tensor,
                     actions: torch.Tensor,
                     log_prob_old: torch.Tensor,
                     rewards: torch.Tensor,
                     adavntages: torch.Tensor) -> None:
        """
        Update the policy and value networks based on PPO loss.

        Args:
            states (torch.Tensor): _description_
            actions (torch.Tensor): _description_
            log_prob_old (torch.Tensor): _description_
            rewards (torch.Tensor): _description_
            adavntages (torch.Tensor): _description_
        """
        loss = self.__compute_loss(states, actions, log_prob_old, rewards, adavntages)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()