"""Critic class and networks for DDPG."""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from mp_rl.core.utils import soft_update, sync_networks, sync_grads


class Critic:
    """Critic class encapsulating the critic and training process for the DDPG critic."""

    def __init__(self, size_s: int, size_a: int, lr: float, grad_clip: float = np.inf):
        """Initialize the critic, the critic network and create its target as an exact copy.

        Args:
            size_s: Critic network input state size. If the input consists of a state and a
                goal, the size is equal to their sum.
            size_a: Critic network input action size.
            lr: Critic network learning rate.
            grad_clip: Gradient clipping value for optimizer steps.
        """
        self.critic_net = CriticNetwork(size_s, size_a)
        self.optim = torch.optim.Adam(self.critic_net.parameters(), lr=lr)
        self.size_s = size_s
        self.size_a = size_a
        self.target_net = CriticNetwork(size_s, size_a)
        self.target_net.load_state_dict(self.critic_net.state_dict())
        self.grad_clip = grad_clip
        self.dist = False

    def __call__(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Run a critic net forward pass.

        Args:
            states: Input states.
            actions: Input actions.

        Returns:
            An action value tensor.
        """
        return self.critic_net(states, actions)

    def target(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute the action value with the target network.

        Args:
            states: Input states.
            actions: Input actions.

        Returns:
            An action value tensor.
        """
        return self.target_net(states, actions)

    def backward_step(self, loss: torch.tensor):
        """Perform a backward pass with an optimizer step.

        Args:
            loss: Critic network loss.
        """
        self.optim.zero_grad()
        loss.backward()
        if self.dist:
            sync_grads(self.critic_net)
        self.optim.step()

    def update_target(self, tau: float):
        """Update the target network with a soft parameter transfer update.

        Args:
            tau: Averaging fraction of the parameter update for the action network.
        """
        soft_update(self.critic_net, self.target_net, tau)

    def init_dist(self):
        """Initialize the critic net as a DDP network and reload the target network."""
        self.dist = True
        sync_networks(self.critic_net)
        # Target reloads state dict because network sync overwrites weights in process rank 1 to n
        # with the weights of action_net from process rank 0
        self.target_net.load_state_dict(self.critic_net.state_dict())


class CriticNetwork(nn.Module):
    """State action critic network for the critic."""

    def __init__(self, size_s: int, size_a: int):
        """Initialize the network.

        Args:
            size_s: Input layer size.
            size_a: Output layer size.
        """
        super().__init__()
        self.l1 = nn.Linear(size_s + size_a, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute the network forward pass.

        Args:
            state: Input state tensor.
            action: Input action tensor.

        Returns:
            The network output.
        """
        x = torch.relu(self.l1(torch.cat([state, action], dim=1)))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))
        return self.l4(x)
