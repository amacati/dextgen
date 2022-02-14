import torch
import torch.nn as nn

from utils import soft_update


class Actor:

    def __init__(self, n_states, n_actions, noise_process, lr, eta):
        self.action_net = ActorNetwork(n_states, n_actions)
        self.optim = torch.optim.Adam(self.action_net.parameters(), lr=lr)
        self.target_net = ActorNetwork(n_states, n_actions)
        self.target_net.load_state_dict(self.action_net.state_dict())
        self.noise_process = noise_process
        self.eta = eta

    def __call__(self, state):
        return self.action_net(state)

    def update_target(self, tau):
        soft_update(self.action_net, self.target_net, tau)


class ActorNetwork(nn.Module):

    def __init__(self, n_states: int, n_actions: int):
        super().__init__()
        self.l1 = nn.Linear(n_states, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))
        return torch.tanh(self.l4(x))
