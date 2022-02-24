"""DDPG class encapsulating the Deep Deterministic Policy Gradient algorithm.

Assumes dictionary gym environments.
"""

import argparse
from typing import List, Union
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
import gym
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from mp_rl.core.utils import unwrap_obs
from mp_rl.core.noise import GaussianNoise
from mp_rl.core.actor import Actor
from mp_rl.core.critic import Critic
from mp_rl.core.normalizer import Normalizer
from mp_rl.core.replay_buffer import HERBuffer, her_sampling, TrajectoryBuffer

T = Union[np.ndarray, torch.Tensor]


class DDPG:
    """Deep Deterministic Policy Gradient algorithm class.

    Uses a state/goal normalizer and the HER sampling method to solve sparse reward environments.
    """

    def __init__(self,
                 env: gym.Env,
                 args: argparse.Namespace,
                 world_size: int = 1,
                 rank: int = 0,
                 dist: bool = False):
        """Initialize the Actor, Critic, HERBuffer and torch's DDP if required.

        Args:
            env: OpenAI dictionary gym environment.
            args: User settings and configs merged into a single namespace.
            world_size: Process group world size for distributed training.
            rank: Process rank for distributed training.
            dist: Toggles distributed training mode.
        """
        self.env = env
        self.args = args
        size_s = len(env.observation_space["observation"].low)
        size_a = len(env.action_space.low)
        size_g = len(env.observation_space["desired_goal"].low)
        # noise_process = OrnsteinUhlenbeckNoise(args.mu, args.sigma, size_a)
        noise_process = GaussianNoise(0, args.sigma, size_a)
        self.actor = Actor(size_s + size_g, size_a, noise_process, args.actor_lr, args.eps,
                           args.action_clip, args.grad_clip)
        self.critic = Critic(size_s + size_g, size_a, args.critic_lr, args.grad_clip)
        self.state_norm = Normalizer(size_s, world_size, clip=args.state_clip)
        self.goal_norm = Normalizer(size_g, world_size, clip=args.goal_clip)
        self.T = env._max_episode_steps
        self.buffer = HERBuffer(size_s,
                                size_a,
                                size_g,
                                self.T,
                                args.k,
                                args.buffer_size,
                                reward_fun=self.env.compute_reward)
        self.action_max = torch.as_tensor(self.env.action_space.high, dtype=torch.float32)
        self.dist = False
        self.world_size = world_size
        self.rank = rank
        self.PATH = Path(__file__).parents[2] / "saves" / self.args.env
        if dist:
            self.init_ddp()

    def train(self):
        """Train a policy to solve the environment with DDPG.

        Trajectories are resampled with HER to solve sparse reward environments. Supports
        distributed training across multiple processes.

        .. DDPG paper:
            https://arxiv.org/pdf/1509.02971.pdf
        """
        if self.rank == 0:
            status_bar = tqdm(total=self.args.epochs, desc="Epochs", position=0, leave=True)
            success_log = tqdm(total=0, position=1, bar_format='{desc}', leave=True)
            ep_success = []
            ep_time = []
        for epoch in range(self.args.epochs):
            epoch_start = time.perf_counter()
            for _ in range(self.args.cycles):
                for _ in range(self.args.rollouts):
                    ep_buffer = self.buffer.get_trajectory_buffer()
                    obs = self.env.reset()
                    state, goal, agoal = unwrap_obs(obs)
                    for t in range(self.T):
                        with torch.no_grad():
                            action = self.actor.select_action(self.wrap_obs(state, goal))
                        next_obs, _, _, _ = self.env.step(action)
                        next_state, _, next_agoal = unwrap_obs(next_obs)
                        ep_buffer.append(state, action, goal, agoal)
                        state, agoal = next_state, next_agoal
                    ep_buffer.append(state, agoal)
                    self.buffer.append(ep_buffer)
                self._update_norm(ep_buffer)
                for _ in range(self.args.batches):
                    self._train_agent()
                self.actor.update_target(self.args.tau)
                self.critic.update_target(self.args.tau)
            epoch_end = time.perf_counter()
            av_success = self.eval_agent()
            if self.rank == 0:
                ep_success.append(av_success)
                ep_time.append(epoch_end - epoch_start)
                success_log.set_description_str("Current success rate: {:.3f}".format(av_success))
                status_bar.update()
                if self.args.save:
                    self.save()
            if av_success > self.args.early_stop:
                if self.rank == 0:
                    self.generate_plots(ep_success, ep_time)
                return
        if self.rank == 0:
            self.generate_plots(ep_success, ep_time)

    def _train_agent(self):
        """Train the agent and critic network with experience sampled from the replay buffer."""
        states, actions, rewards, next_states, goals = self.buffer.sample(self.args.batch_size)
        obs_T, obs_next_T = self.wrap_obs(states, goals), self.wrap_obs(next_states, goals)
        actions_T = torch.as_tensor(actions, dtype=torch.float32)
        rewards_T = torch.as_tensor(rewards, dtype=torch.float32)
        with torch.no_grad():
            next_actions_T = self.actor.target(obs_next_T)
            next_q_T = self.critic.target(obs_next_T, next_actions_T)
            next_q_T.detach()  # TODO: remove?
            rewards_T = rewards_T + self.args.gamma * next_q_T  # No dones in fixed length episode
            rewards_T.detach()  # TODO: remove?
            # Clip to minimum reward possible, geometric sum from 0 to inf with gamma and -1 rewards
            torch.clip(rewards_T, -1 / (1 - self.args.gamma), 0, out=rewards_T)
        q_T = self.critic(obs_T, actions_T)
        critic_loss_T = (rewards_T - q_T).pow(2).mean()
        actions_T = self.actor(obs_T)
        next_q_T = self.critic(obs_T, actions_T)
        action_norm_T = self.args.action_norm * (actions_T / self.action_max).pow(2).mean()
        actor_loss_T = -next_q_T.mean() + action_norm_T
        self.actor.backward_step(actor_loss_T)
        self.critic.backward_step(critic_loss_T)

    def _update_norm(self, ep_buffer: TrajectoryBuffer):
        """Update the normalizers with an episode of play experience.

        Samples the trajectory instead of taking every experience to create a goal distribution that
        is equal to what the networks encouter.

        Args:
            ep_buffer: Buffer containing a trajectory of replay experience.

        Todo:
            It is unclear if sampling the goals from HER is necessary, or if the unmodified
            trajectory can be used. This should be investigated and simplified if possible.
        """
        buffers = [np.expand_dims(ep_buffer[x], axis=0) for x in ("s", "a", "g", "ag")]
        states, _, _, _, goals = her_sampling(*buffers, self.T, self.buffer.p_her,
                                              self.env.compute_reward)
        self.state_norm.update(states)
        self.goal_norm.update(goals)

    def eval_agent(self) -> float:
        """Evaluate the current agent performance on the gym task.

        Runs `args.evals` times and averages the success rate. If distributed training is enabled,
        further averages over all distributed evaluations.
        """
        self.actor.eval()
        success = 0
        for _ in range(self.args.evals):
            state, goal, _ = unwrap_obs(self.env.reset())
            for t in range(self.T):
                with torch.no_grad():
                    action = self.actor.select_action(self.wrap_obs(state, goal))
                next_obs, _, _, info = self.env.step(action)
                state, goal, _ = unwrap_obs(next_obs)
            success += info["is_success"]
        self.actor.train()
        if self.dist:
            success_rate = torch.tensor([success / self.args.evals], dtype=torch.float32)
            dist.all_reduce(success_rate)  # In-place op
            return success_rate.item() / self.world_size
        return success / self.args.evals

    def save(self):
        """Save the actor network and the normalizers for testing and inference.

        Saves are located under `/save/<env_name>/`.
        """
        if not self.PATH.is_dir():
            self.PATH.mkdir(parents=True, exist_ok=True)
        if self.dist:
            torch.save(self.actor.action_net.module.state_dict(), self.PATH / "actor.pt")
        else:
            torch.save(self.actor.action_net.state_dict(), self.PATH / "actor.pt")
        self.state_norm.save(self.PATH / "state_norm.pkl")
        self.goal_norm.save(self.PATH / "goal_norm.pkl")

    def generate_plots(self, ep_success: List[float], ep_time: List[float]):
        """Generate and save a plot from training statistics.

        Saves are located under `/save/<env_name>/`.

        Args:
            ep_success: Episode agent evaluation success rate.
            ep_time: Episode training process time. Excludes the evaluation timing.
        """
        if not self.PATH.is_dir():
            self.PATH.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(15, 4))
        ax[0].plot(ep_success)
        ax[0].set_xlabel('Episode')
        ax[0].set_ylabel('Success rate')
        ax[0].set_title('Agent performance over time')
        ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))

        ax[1].plot(ep_time)
        ax[1].set_xlabel('Episode')
        ax[1].set_ylabel('Time in s')
        ax[1].set_title('Episode compute times')
        ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(self.PATH / "stats.png")

    def wrap_obs(self, states: np.ndarray, goals: np.ndarray) -> torch.Tensor:
        """Wrap states and goals into a contingent input tensor.

        Args:
            states: Input states array.
            goals: Input goals array.

        Returns:
            A fused state goal tensor.
        """
        states, goals = self.state_norm(states), self.goal_norm(goals)
        x = np.concatenate((states, goals), axis=states.ndim - 1)
        return torch.as_tensor(x, dtype=torch.float32)

    def init_ddp(self):
        """Configure actor, critic and normalizers for distributed training."""
        self.actor.init_ddp()
        self.critic.init_ddp()
        self.state_norm.init_ddp()
        self.goal_norm.init_ddp()
        self.dist = True
