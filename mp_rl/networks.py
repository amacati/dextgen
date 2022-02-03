import torch
import torch.nn as nn


class DDPGActor(nn.Module):
    
    def __init__(self, n_states: int, n_actions: int):
        super().__init__()
        self.l1 = nn.Linear(n_states, 400)
        self.l2 = nn.Linear(400, 200)
        self.l3 = nn.Linear(200, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return torch.tanh(self.l3(x))

    
class DDPGCritic(nn.Module):
    
    def __init__(self, n_states: int, n_actions: int):
        super().__init__()
        self.l1 = nn.Linear(n_states, 400)
        self.l2 = nn.Linear(400+n_actions, 200)
        self.l3 = nn.Linear(200, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state = torch.relu(self.l1(state))
        x = torch.relu(self.l2(torch.cat([state, action], dim=1)))
        return self.l3(x)


class InputNorm(nn.Module):
    
    def __init__(self, size: int, tau: float = 0.01):
        super().__init__()
        self._tau = tau
        self._ntau = 1 - tau
        self._init = False
        self._eps = 1e-5
        input_mean = torch.zeros(size)
        self.input_mean = nn.Parameter(input_mean, requires_grad=False)
        self._input_smean = torch.zeros(size)
        input_var = torch.ones(size)
        self.input_var = nn.Parameter(input_var, requires_grad=False)

    def _update_input_dist(self, x):
        with torch.no_grad():
            if not self._init:
                self.input_mean.copy_(torch.mean(x, 0))
                self._input_smean = torch.mean(torch.square(x), 0)
                self.input_var.copy_(torch.var(x, 0))
                self._init = True
            else:
                input_mean = self.input_mean * self._ntau + torch.sum(x, 0) * self._tau
                self.input_mean.copy_(input_mean)
                self._input_smean = self._input_smean * self._ntau + torch.sum(torch.square(x), 0) * self._tau
                input_var = self._input_smean - torch.square(self.input_mean)
                self.input_var.copy_(input_var)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._update_input_dist(x)
        return (x - self.input_mean)/(self.input_var + self._eps)  # Add eps for numerical stability