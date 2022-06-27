from pathlib import Path

import gym
import torch

import envs
from mp_rl.core.utils import unwrap_obs
from mp_rl.core.actor import DDP

if __name__ == "__main__":
    root = Path(__file__).parents[1] / "saves" / "FlatPJCube-v0"
    env = gym.make("FlatPJCube-v0")

    obs = env.reset()
    state, goal, achieved_goal = unwrap_obs(obs)

    actor_net = DDP(len(state) + len(goal), 10, 4, 256)
    actor_net.load_state_dict(torch.load(root / "actor.pt"))
    input_T = torch.concat((torch.as_tensor(state), torch.as_tensor(goal)))
    print(goal)
    input_T = input_T.float()
    input_T.requires_grad = True

    action = actor_net(input_T)
    print(action)
    print(torch.autograd.grad(action[0], input_T))
