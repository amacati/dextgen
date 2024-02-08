"""The ``actor`` module contains the actor class as well as the actor networks.

The :class:`.Actor` acts as a wrapper around the actual deterministic policy network to provide
action selection and loading utilities.

:class:`.DDP` is a vanilla deep deterministic policy network implementation.

:class:`.PosePolicyNet` implements an actor network with three output heads, one for translation
control, a second for rotation control and a third for gripper control. It uses a special
internal orientation representation that smoothly varies in SO(3). See `On the Continuity of
Rotation Representations in Neural Networks <https://ieeexplore.ieee.org/document/8953486>`_ for
details.
"""
from __future__ import annotations
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from mp_rl.core.utils import soft_update
from mp_rl.utils import import_guard

if import_guard():
    from mp_rl.core.noise import NoiseProcess  # noqa: TC001, is guarded


class Actor:
    """Actor class encapsulating the action selection and training process for the DDPG actor."""

    def __init__(self,
                 size_s: int,
                 size_a: int,
                 nlayers: int,
                 layer_width: int,
                 noise_process: NoiseProcess,
                 lr: float,
                 eps: float,
                 action_clip: float = np.Inf,
                 grad_clip: float = np.Inf):
        """Initialize the actor, the actor networks and create its target as an exact copy.

        Args:
            actor_net_type: Deterministic action network type. Either ``PosePolicyNet`` or ``DDP``.
            size_s: Actor network input state size. If the input consists of a state and a
                goal, the size is equal to their sum.
            size_a: Actor network output action size.
            nlayers: Number of network layers.
            layer_width: Number of nodes per layer. Does not influence input and output size.
            noise_process: Noise process to sample action noise from.
            lr: Actor network learning rate.
            eps: Random action probability during training.
            action_clip: Action output clipping value.
            grad_clip: Gradient clipping value for optimizer steps.
        """
        self.action_net = DDP(size_s, size_a, nlayers, layer_width)
        self.optim = torch.optim.Adam(self.action_net.parameters(), lr=lr)
        self.target_net = DDP(size_s, size_a, nlayers, layer_width)
        self.target_net.load_state_dict(self.action_net.state_dict())
        self.noise_process = noise_process
        self.eps = eps
        self.action_clip = action_clip
        self.grad_clip = grad_clip
        self._train = True

    def select_action(self, states: torch.Tensor) -> np.ndarray:
        """Select an action for the given input states.

        If in train mode, samples noise and chooses completely random actions with probability
        `self.eps`. If in evaluation mode, only clips the action to the maximum value.

        Note:
            This function returns a numpy array instead of a torch.Tensor for compatibility with the
            gym environments.

        Args:
            states: Input states.

        Returns:
            A numpy array of actions.
        """
        if self._train:  # With noise process + random sampling for exploration
            actions = self.action_net(states).numpy()
            if np.random.rand() < self.eps:
                actions = self.noise_process.sample()
            else:
                actions += np.random.normal(0, 0.2, actions.shape)
            np.clip(actions, -self.action_clip, self.action_clip, out=actions)  # In-place op
        else:  # No random exploration moves
            actions = self.action_net(states).numpy()
            np.clip(actions, -self.action_clip, self.action_clip, out=actions)
        return actions

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        """Run a forward pass directly on the action net.

        Args:
            states: Input states.

        Returns:
            An action tensor.
        """
        return self.action_net(states)

    def backward_step(self, loss: torch.Tensor):
        """Perform a backward pass with an optimizer step.

        Args:
            loss: Actor network loss.
        """
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def target(self, states: torch.Tensor) -> torch.Tensor:
        """Compute actions with the target network and without noise.

        Args:
            states: Input states.

        Returns:
            An action tensor.
        """
        return self.target_net(states)

    def eval(self):
        """Set the actor to eval mode without noise in the action selection."""
        self._train = False
        self.action_net.eval()

    def train(self):
        """Set the actor to train mode with noisy actions."""
        self._train = True
        self.action_net.train()

    def update_target(self, tau: float):
        """Update the target network with a soft parameter transfer update.

        Args:
            tau: Averaging fraction of the parameter update for the action network.
        """
        soft_update(self.action_net, self.target_net, tau)

    def load(self, path: Path):
        """Load saved network weights for the actor and take care of syncronizations.

        Args:
            path: Path to the saved state dict.
        """
        self.action_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.action_net.state_dict())


class DDP(nn.Module):
    """Continuous action choice network for the agent."""

    def __init__(self, size_s: int, size_a: int, nlayers: int, layer_width: int):
        """Initialize the network.

        Args:
            size_s: Input layer size.
            size_a: Output layer size.
            nlayers: Number of network layers.
            layer_width: Number of nodes per layer. Does not influence input and output size.
        """
        assert nlayers >= 1
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(nlayers):
            size_in = size_s if i == 0 else layer_width
            size_out = size_a if i == nlayers - 1 else layer_width
            self.layers.append(nn.Linear(size_in, size_out))
            activation = nn.Tanh() if i == nlayers - 1 else nn.ReLU()
            self.layers.append(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the network forward pass.

        Args:
            x: Input tensor.

        Returns:
            The network output.
        """
        for layer in self.layers:
            x = layer(x)
        return x
