"""The ``critic`` module contains the critic class as well as the critic network.

The :class:`.Critic` acts as a wrapper around the actual critic Q-function to provide distributed
training support and loading utilities.

:class:`.CriticNetwork` is a vanilla deep state-action network implementation.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from mp_rl.core.utils import soft_update, sync_networks, sync_grads
from mp_rl.utils import import_guard

if import_guard():
    from pathlib import Path  # noqa: TC003, is guarded


class Critic:
    """Critic class encapsulating the critic and training process for the DDPG critic."""

    def __init__(self,
                 size_s: int,
                 size_a: int,
                 nlayers: int,
                 layer_width: int,
                 lr: float,
                 grad_clip: float = np.inf):
        """Initialize the critic, the critic network and create its target as an exact copy.

        Args:
            size_s: Critic network input state size. If the input consists of a state and a
                goal, the size is equal to their sum.
            size_a: Critic network input action size.
            nlayers: Number of network layers.
            layer_width: Number of nodes per layer. Does not influence input and output size.
            lr: Critic network learning rate.
            grad_clip: Gradient clipping value for optimizer steps.
        """
        self.critic_net = CriticNetwork(size_s, size_a, nlayers, layer_width)
        self.optim = torch.optim.Adam(self.critic_net.parameters(), lr=lr)
        self.size_s = size_s
        self.size_a = size_a
        self.target_net = CriticNetwork(size_s, size_a, nlayers, layer_width)
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
        """Synchronize the critic network across MPI workers and reload the target network."""
        self.dist = True
        sync_networks(self.critic_net)
        # Target reloads state dict because network sync overwrites weights in process rank 1 to n
        # with the weights of action_net from process rank 0
        self.target_net.load_state_dict(self.critic_net.state_dict())

    def load(self, path: Path):
        """Load saved network weights for the critic and take care of syncronizations.

        Args:
            path: Path to the saved state dict.
        """
        self.critic_net.load_state_dict(torch.load(path))
        if self.dist:
            sync_networks(self.critic_net)
        self.target_net.load_state_dict(self.critic_net.state_dict())


class CriticNetwork(nn.Module):
    """State action critic network for the critic."""

    def __init__(self, size_s: int, size_a: int, nlayers: int, layer_width: int):
        """Initialize the network.

        Args:
            size_s: Input layer size.
            size_a: Output layer size.
            nlayers: Number of network layers.
            layer_width: Number of nodes per layer. Does not influence input and output size.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(nlayers):
            size_in = size_s + size_a if i == 0 else layer_width
            size_out = 1 if i == nlayers - 1 else layer_width
            self.layers.append(nn.Linear(size_in, size_out))
            if i != nlayers - 1:  # Last layer doesn't get an activation function
                self.layers.append(nn.ReLU())

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute the network forward pass.

        Args:
            state: Input state tensor.
            action: Input action tensor.

        Returns:
            The network output.
        """
        x = torch.cat([state, action], dim=1)
        for layer in self.layers:
            x = layer(x)
        return x
