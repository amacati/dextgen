import torch
import torch.nn as nn
import numpy as np

from mp_rl.core.utils import soft_update


class Actor:

    def __init__(self, size_s, size_a, noise_process, lr, eps, action_clip: float = np.Inf,
                 grad_clip: float = np.Inf):
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

    def select_action(self, states):
        actions = self.action_net(states).numpy()
        if self._train:  # With noise process + random sampling for exploration
            actions += self.noise_process.sample()
            np.clip(actions, -self.action_clip, self.action_clip, out=actions)  # In-place op
            random_actions = np.random.uniform(-self.action_clip, self.action_clip, actions.shape)
            choice = np.random.binomial(1, self.eps, 1)[0]
            actions += choice*(random_actions - actions)
        else:  # No random exploration moves
            np.clip(actions, -self.action_clip, self.action_clip, out=actions)
        return actions

    def __call__(self, states):
        return self.action_net(states)

    def backward(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def target(self, states):
        return self.target_net(states)

    def eval(self):
        self._train = False
        self.action_net.eval()

    def train(self):
        self._train = True
        self.action_net.train()

    def update_target(self, tau):
        soft_update(self.action_net, self.target_net, tau)

    @property
    def dist(self):
        return self._dist

    @dist.setter
    def dist(self, value: bool):
        if value:
            raise NotImplementedError("Dist currently not supported")
        self._dist = value


class ActorNetwork(nn.Module):

    def __init__(self, size_s: int, size_a: int):
        super().__init__()
        self.l1 = nn.Linear(size_s, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, size_a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))
        return torch.tanh(self.l4(x))
