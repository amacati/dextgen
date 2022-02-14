from typing import Union
from functools import singledispatchmethod

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from mp_rl.core.utils import soft_update


T = Union[np.ndarray, torch.Tensor]


class DDPG:

    def __init__(self, actor, actor_target, critic, critic_target, actor_lr, critic_lr, tau, gamma,
                 noise_process, action_clip=None, actor_clip=None, critic_clip=None):
        self.actor = Actor()
        self.critic = Critic()
        self.actor_target = actor_target
        self.critic = critic
        self.critic_target = critic_target
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=0.01)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=0.01)
        self.noise_process = noise_process
        self.action_clip = action_clip or (-np.Inf, np.Inf)
        self.actor_clip = actor_clip or np.Inf
        self.critic_clip = critic_clip or np.Inf
        self.explore = True
        self.tau = tau
        self.gamma = gamma
        self.actor_path = None
        self.critic_path = None

    def init_ddp(self):
        self.actor = DDP(self.actor)
        self.critic = DDP(self.critic)
        self.actor_target = DDP(self.actor_target)
        self.critic_target = DDP(self.critic_target)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr, weight_decay=0.01)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr, weight_decay=0.01)  # noqa: E501

    def train(self):
        if self.rank == 0:
            status_bar = tqdm(total=config["epochs"]*config["cycles"], desc="Training iterations",
                            position=0, leave=False)
            reward_log = tqdm(total=0, position=1, bar_format='{desc}', leave=False)
            ep_success = []
        for epoch in range(self.args.n_epochs):
            for _ in range(self.args.n_cycles):
                for _ in range(self.args.n_rollouts):
                    ep_buffer = self.buffer.get_trajectory_buffer()
                    obs = self.env.reset()
                    state, goal, agoal = unwrap_obs(obs)
                    for t in range(self.T):
                        with torch.no_grad():
                            action = self.actor(state, goal)
                        next_obs, _, _, info = self.env.step(action)
                        next_state, _, next_agoal = unwrap_obs(next_obs)
                        ep_buffer.append(state, action, goal, agoal)
                        state = next_state
                        agoal = next_agoal
                    self.buffer.append(ep_buffer)
                for _ in range(self.args.n_train):
                    self.train_agent()
            av_success = self.eval_agent()
            if self.rank == 0:
                ep_success.append(av_success)
                success_log.set_description_str("Current success rate: {:.1f}".format(av_success))  # noqa: E501
                status_bar.update()
            if av_success > self.early_stop:
                return

    def action(self, states: T) -> np.ndarray:
        states = self.sanitize_array(states)
        with torch.no_grad():
            noise = self.noise_process.sample() if self.explore else 0
            actions = self.actor(states).cpu().numpy() + noise
            if self.action_clip is not None:
                actions = np.clip(actions, self.action_clip[0], self.action_clip[1])
        return actions

    def train_actor(self, train_batch: list[T]):
        states, actions = [self.sanitize_array(x) for x in train_batch[:2]]
        self.actor_optim.zero_grad()
        actions = self.actor(states)
        next_q = self.critic(states, actions)
        (-torch.mean(next_q)).backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_clip)
        self.actor_optim.step()

    def train_critic(self, train_batch: list[T]):
        states, actions, rewards, next_states, dones = [self.sanitize_array(x) for x in train_batch]
        self.critic_optim.zero_grad()
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_q = self.critic_target(next_states, next_actions)
            rewards = rewards + self.gamma * next_q  # No dones in fixed length episode
        q_actions = self.critic(states, actions)
        nn.functional.mse_loss(q_actions, rewards).backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_clip)
        self.critic_optim.step()

    def update_targets(self):
        self.actor_target = soft_update(self.actor, self.actor_target, self.tau)
        self.critic_target = soft_update(self.critic, self.critic_target, self.tau)

    def save(self, actor_path, critic_path, config_path):
        if isinstance(self.actor, DDP):  # Unwrap module from DDP
            torch.save(self.actor.module, actor_path)
            torch.save(self.critic.module, critic_path)
        else:
            torch.save(self.actor, actor_path)
            torch.save(self.critic, critic_path)
        self.actor_path = actor_path
        self.critic_path = critic_path
        config = {"actor_lr": self.actor_lr, "critic_lr": self.critic_lr,
                  "noise_process": self.noise_process, "action_clip": self.action_clip,
                  "actor_clip": self.actor_clip, "critic_clip":  self.critic_clip,
                  "explore": self.explore, "tau": self.tau, "gamma": self.gamma,
                  "actor_path": self.actor_path, "critic_path": self.critic_path}
        with open(config_path, "wb") as f:
            pickle.dump(config, f)

    def load_models(self, actor_path, critic_path):
        self.actor = torch.load(self.actor_path)
        self.actor_target = torch.load(self.actor_path)
        self.critic = torch.load(self.critic_path)
        self.critic_target = torch.load(self.critic_path)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr, weight_decay=0.01)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr, weight_decay=0.01)  # noqa: E501

    @singledispatchmethod
    def sanitize_array(self, x):
        raise TypeError(f"Cannot process array inputs of type {type(x)}")

    @sanitize_array.register
    def _(self, x: np.ndarray):
        if x.ndim == 1:
            return torch.unsqueeze(torch.as_tensor(x, dtype=torch.float32), 1)
        elif x.ndim == 2:
            return torch.as_tensor(x, dtype=torch.float32)
        raise ValueError("Array shape exceeds expected dimensions of 1 or 2.")

    @sanitize_array.register
    def _(self, x: torch.Tensor):
        if x.ndim == 1:
            return torch.unsqueeze(x, 1).float()
        elif x.ndim == 2:
            return x.float()
        raise ValueError("Tensor shape exceeds expected dimensions of 1 or 2")


def load_ddpg(config_path):
    ddpg = DDPG(DDPGActor(1, 1), DDPGActor(1, 1), DDPGCritic(1, 1), DDPGCritic(1, 1),
                1., 1., 1., 1., None)  # Values get overwritten
    with open(config_path, "rb") as f:
        config = pickle.load(f)
    for key, val in config.items():
        setattr(ddpg, key, val)
    ddpg.load_models(ddpg.actor_path, ddpg.critic_path)
    return ddpg
