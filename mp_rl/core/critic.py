import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from mp_rl.core.utils import soft_update


class Critic:

    def __init__(self, size_s, size_a, lr, grad_clip=np.inf):
        self.critic_net = CriticNetwork(size_s, size_a)
        self.optim = torch.optim.Adam(self.critic_net.parameters(), lr=lr)
        self.size_s = size_s
        self.size_a = size_a
        self.target_net = CriticNetwork(size_s, size_a)
        self.target_net.load_state_dict(self.critic_net.state_dict())
        self.grad_clip = grad_clip
        self.dist = False

    def __call__(self, states, actions):
        return self.critic_net(states, actions)

    def update_target(self, tau):
        soft_update(self.critic_net, self.target_net, tau)

    def target(self, states, actions):
        return self.target_net(states, actions)

    def backward(self, loss: torch.tensor):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def init_ddp(self):
        self.critic_net = DDP(self.critic_net)
        # Target reloads state dict because DDP overwrites weights in process rank 1 to n with the
        # weights of action_net from process rank 0
        self.target_net.load_state_dict(self.critic_net.module.state_dict())


class CriticNetwork(nn.Module):

    def __init__(self, size_s: int, size_a: int):
        super().__init__()
        self.l1 = nn.Linear(size_s + size_a, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, 1)

    def forward(self, state: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.l1(torch.cat([state, action], dim=1)))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))
        return self.l4(x)
