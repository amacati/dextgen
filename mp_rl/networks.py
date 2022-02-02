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
