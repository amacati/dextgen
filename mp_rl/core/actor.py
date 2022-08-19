"""The ``actor`` module contains the actor class as well as the actor networks.

The :class:`.Actor` acts as a wrapper around the actual deterministic policy network to provide
action selection, distributed training support and loading utilities.

:class:`.DDP` is a vanilla deep deterministic policy network implementation.

:class:`.PosePolicyNet` implements an actor network with three output heads, one for translation
control, a second for rotation control and a third for gripper control. It uses a special
internal orientation representation that smoothly varies in SO(3). See `On the Continuity of
Rotation Representations in Neural Networks <https://ieeexplore.ieee.org/document/8953486>`_ for
details.
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple
from uuid import uuid4

import torch
import torch.nn as nn
import numpy as np

from mp_rl.core.utils import soft_update, sync_networks, sync_grads
from mp_rl.utils import import_guard

if import_guard():
    from mp_rl.core.noise import NoiseProcess  # noqa: TC001, is guarded


class Actor:
    """Actor class encapsulating the action selection and training process for the DDPG actor."""

    def __init__(self,
                 actor_net_type: str,
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
        NetClass = DDP if actor_net_type == "DDP" else PosePolicyNet
        self.action_net = NetClass(size_s, size_a, nlayers, layer_width)
        self.optim = torch.optim.Adam(self.action_net.parameters(), lr=lr)
        self.target_net = NetClass(size_s, size_a, nlayers, layer_width)
        self.target_net.load_state_dict(self.action_net.state_dict())
        self.noise_process = noise_process
        self.eps = eps
        self.action_clip = action_clip
        self.grad_clip = grad_clip
        self._train = True
        self.dist = False

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
        if self.dist:
            sync_grads(self.action_net)
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

    def init_dist(self):
        """Synchronize the actor network across MPI nodes and reload the target network."""
        self.dist = True
        sync_networks(self.action_net)
        # Target reloads state dict because network sync overwrites weights in process rank 1 to n
        # with the weights of action_net from process rank 0
        self.target_net.load_state_dict(self.action_net.state_dict())

    def load(self, path: Path):
        """Load saved network weights for the actor and take care of syncronizations.

        Args:
            path: Path to the saved state dict.
        """
        self.action_net.load_state_dict(torch.load(path))
        if self.dist:
            sync_networks(self.action_net)
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


class PosePolicyNet(nn.Module):
    """Continuous action choice network for the agent.

    Uses a special embedding for pose actions.
    """

    _idx_select = torch.IntTensor([0, 3, 6, 1, 4, 7, 2, 5, 8])  # Permutation indices for embedding

    def __init__(self, size_s: int, size_a: int, nlayers: int, layer_width: int):
        """Initialize the network.

        We implicitly assume that the first 3 outputs are position outputs x, y, z and the next 9
        outputs are combined into a rotation matrix. Our network learns an intermediate
        representation according to https://arxiv.org/pdf/1812.07035.pdf which is mapped back to
        the original parameterization of SO(3).

        Args:
            size_s: Input layer size.
            size_a: Output layer size.
            nlayers: Number of network layers.
            layer_width: Number of nodes per layer. Does not influence input and output size.
        """
        assert size_a > 9
        assert nlayers >= 1
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(nlayers):
            size_in = size_s if i == 0 else layer_width
            size_out = size_a - 3 if i == nlayers - 1 else layer_width
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
        assert not torch.isnan(x).any(), f"NaN values in forward. Input was {x}"
        for layer in self.layers:
            x = layer(x)

        # Map the 6D rotation representation in x[3:9] to the original SO3 representation. The
        # following code uses the naming convention of https://arxiv.org/pdf/1812.07035.pdf. We
        # interpret x_rot to be [a11, a12, a13, a21, a22, a23]
        flatrotmat = self.embedding2flatmat_safe(x[..., 3:9])
        return torch.concat((x[..., :3], flatrotmat, x[..., 9:]), dim=-1)

    def forward_include_network_output(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the forward pass and include the input before the embedding in the output.

        Useful for regularizing the tanh output of the embedding. Tanh can saturate and lead to
        diverent policies.

        Args:
            x: Input tensor.

        Returns:
            The network output and the network output before transforming the embedding into a
                rotation matrix.
        """
        assert not torch.isnan(x).any(), f"NaN values in forward. Input was {x}"
        for layer in self.layers:
            x = layer(x)
        flatrotmat = self.embedding2flatmat_safe(x[..., 3:9])
        return torch.concat((x[..., :3], flatrotmat, x[..., 9:]), dim=-1), x

    @staticmethod
    def embedding2flatmat(embedding: torch.Tensor) -> torch.Tensor:
        """Transform a 6D rotation embedding into a proper rotation matrix from SO(3).

        Args:
            embedding: A tensor of shape (..., 6) with values in [-1, 1].

        Returns:
            A tensor of rotation matrices with shape (..., 9). Matrices are ordered such that a
            reshape to (3, 3) will yield the correct matrices without transposing.
        """
        assert embedding.shape[-1] == 6
        b1 = embedding[..., 0:3] / torch.linalg.norm(embedding[..., 0:3], dim=-1, keepdim=True)
        # Torch sum for batched dot product
        b2 = embedding[..., 3:6] - (torch.sum(b1 * embedding[..., 3:6], dim=-1, keepdim=True) * b1)
        b2 = b2 / torch.linalg.norm(b2, dim=-1, keepdim=True)
        b3 = torch.cross(b1, b2, dim=-1)
        flatrotmat = torch.concat((b1, b2, b3), dim=-1)
        if torch.isnan(flatrotmat).any():
            path = Path(__file__).parents[2] / (str(uuid4()) + "_nan_embedding.pt")
            torch.save(embedding, path)
        assert not torch.isnan(flatrotmat).any(), f"NaN values in flatrotmat. Input was {embedding}"

        # We have to permute the elements in order to get the correct rotation matrix when reshaping
        return flatrotmat.index_select(-1, PosePolicyNet._idx_select)

    @staticmethod
    def embedding2flatmat_safe(embedding: torch.Tensor) -> torch.Tensor:
        """Transform a 6D rotation embedding into a proper rotation matrix from SO(3).

        Safe, less performant version of the transformation. If b1 is the zero vector, it gets
        replaced by a random vector. If b2 is parallel to b1 it also gets replaces with a random
        vector.

        Args:
            embedding: A tensor of shape (..., 6) with values in [-1, 1].

        Returns:
            A tensor of rotation matrices with shape (..., 9). Matrices are ordered such that a
            reshape to (3, 3) will yield the correct matrices without transposing.
        """
        assert embedding.shape[-1] == 6
        # Extract the first rotation matrix vector. If the vector is 0, fill with a random vector.
        b1 = embedding[..., 0:3]
        b1norm = torch.linalg.norm(b1, dim=-1, keepdim=True)
        b1mask = torch.repeat_interleave(b1norm == 0, 3, dim=-1)
        b1 = torch.where(b1mask, torch.rand(b1.shape), b1)
        b1norm = torch.linalg.norm(b1, dim=-1, keepdim=True)
        b1 = b1 / b1norm
        # Construct the second rotation matrix vector. If the vector is parallel to b1, fill with a
        # random vector.
        b2 = embedding[..., 3:6] - (torch.sum(b1 * embedding[..., 3:6], dim=-1, keepdim=True) * b1)
        b2norm = torch.linalg.norm(b2, dim=-1, keepdim=True)
        b2mask = torch.repeat_interleave(b2norm == 0, 3, dim=-1)
        b2 = torch.where(b2mask, torch.rand(b2.shape), b2)
        b2 = b2 - (torch.sum(b1 * b2, dim=-1, keepdim=True) * b1)
        b2norm = torch.linalg.norm(b2, dim=-1, keepdim=True)  # Calculate new norm and renormalize
        b2 = b2 / b2norm
        b3 = torch.cross(b1, b2, dim=-1)
        b3 = b3 / torch.linalg.norm(b3, dim=-1, keepdim=True)
        flatrotmat = torch.concat((b1, b2, b3), dim=-1)
        if torch.isnan(flatrotmat).any():
            path = Path(__file__).parents[2] / (str(uuid4()) + "_nan_embedding.pt")
            torch.save(embedding, path)
        assert not torch.isnan(flatrotmat).any(), f"NaN values in flatrotmat. Input was {embedding}"
        # We have to permute the elements in order to get the correct rotation matrix when reshaping
        return flatrotmat.index_select(-1, PosePolicyNet._idx_select)
