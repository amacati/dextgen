"""Actor class and networks for DDPG."""

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

from mp_rl.core.utils import soft_update
from mp_rl.core.noise import NoiseProcess


class Actor:
    """Actor class encapsulating the action selection and training process for the DDPG actor."""

    def __init__(self,
                 size_s: int,
                 size_a: int,
                 noise_process: NoiseProcess,
                 lr: float,
                 eps: float,
                 action_clip: float = np.Inf,
                 grad_clip: float = np.Inf):
        """Initialize the actor, the actor networks and create its target as an exact copy.

        Args:
            size_s: Actor network input state size. If the input consists of a state and a
                goal, the size is equal to their sum.
            size_a: Actor network output action size.
            noise_process (NoiseProcess): Noise process to sample action noise from.
            lr: Actor network learning rate.
            eps: Random action probability during training.
            action_clip: Action output clipping value.
            grad_clip: Gradient clipping value for optimizer steps.
        """
        self.action_net = ActorNetwork(size_s, size_a)
        self.optim = torch.optim.Adam(self.action_net.parameters(), lr=lr)
        self.target_net = ActorNetwork(size_s, size_a)
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
        ``. If in evaluation mode, only clips the action to the maximum value.

        Note:
            This function returns a numpy array instead of a torch.Tensor for compatibility with the
            gym environments.

        Args:
            states: Input states.

        Returns:
            A numpy array of actions.
        """
        actions = self.action_net(states).numpy()
        if self._train:  # With noise process + random sampling for exploration
            actions += self.noise_process.sample()
            np.clip(actions, -self.action_clip, self.action_clip, out=actions)  # In-place op
            random_actions = np.random.uniform(-self.action_clip, self.action_clip, actions.shape)
            choice = np.random.binomial(1, self.eps, 1)[0]  # TODO: change
            actions += choice * (random_actions - actions)
        else:  # No random exploration moves
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

    def init_ddp(self):
        """Initialize the action net as a DDP network and reload the target network."""
        self.dist = True
        self.action_net = DDP(self.action_net)
        # Target reloads state dict because DDP overwrites weights in process rank 1 to n with the
        # weights of action_net from process rank 0
        self.target_net.load_state_dict(self.action_net.module.state_dict())


class ActorNetwork(nn.Module):
    """Continuous action choice network for the agent."""

    def __init__(self, size_s: int, size_a: int):
        """Initialize the network.

        Args:
            size_s: Input layer size.
            size_a: Output layer size.
        """
        super().__init__()
        self.l1 = nn.Linear(size_s, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, size_a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the network forward pass.

        Args:
            x: Input tensor.

        Returns:
            The network output.
        """
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))
        return torch.tanh(self.l4(x))
