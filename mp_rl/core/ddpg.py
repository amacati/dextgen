"""``DDPG`` module encapsulating the Deep Deterministic Policy Gradient algorithm.

:class:`.DDPG` initializes the actor, critic, normalizers and noise processes, manages the
synchronization between MPI nodes and takes care of checkpoints during training as well as network
loading if starting from pre-trained networks. It assumes dictionary gym environments.
"""

import argparse
import logging
from typing import List, Optional
from pathlib import Path
import time
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm
import gym
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpi4py import MPI
import json

from mp_rl.core.utils import unwrap_obs
from mp_rl.core.noise import UniformNoise, GaussianNoise, OrnsteinUhlenbeckNoise
from mp_rl.core.actor import Actor, PosePolicyNet
from mp_rl.core.critic import Critic
from mp_rl.core.normalizer import Normalizer
from mp_rl.core.replay_buffer import HERBuffer, TrajectoryBuffer

logger = logging.getLogger(__name__)


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
        """Initialize the Actor, Critic, HERBuffer and activate MPI synchronization if required.

        Args:
            env: OpenAI dictionary gym environment.
            args: User settings and configs merged into a single namespace.
            world_size: Process group world size for distributed training.
            rank: Process rank for distributed training.
            dist: Toggles distributed training mode.
        """
        self.env = env
        # Rewards are calculated from HER buffer. Disable computation for runtime improvement
        self.env.use_step_reward(False)
        self.args = args
        size_s = len(env.observation_space["observation"].low)
        size_a = len(env.action_space.low)
        size_g = len(env.observation_space["desired_goal"].low)
        if args.noise_process == "Uniform":
            noise_process = UniformNoise(size_a)
        elif args.noise_process == "Gaussian":
            noise_process = GaussianNoise(size_a, *args.noise_process_params)
        elif args.noise_process == "OrnsteinUhlenbeck":
            noise_process = OrnsteinUhlenbeckNoise(size_a, *args.noise_process_params)
        else:
            raise argparse.ArgumentError("Required argument 'noise_process' is missing")
        self.actor = Actor(args.actor_net_type, size_s + size_g, size_a, args.actor_net_nlayers,
                           args.actor_net_layer_width, noise_process, args.actor_lr, args.eps,
                           args.action_clip, args.grad_clip)
        self.critic = Critic(size_s + size_g, size_a, args.critic_net_nlayers,
                             args.critic_net_layer_width, args.critic_lr, args.grad_clip)
        state_norm_idx = getattr(args, "state_norm_idx", None)
        goal_norm_idx = getattr(args, "goal_norm_idx", None)
        self.state_norm = Normalizer(size_s, world_size, clip=args.state_clip, idx=state_norm_idx)
        self.goal_norm = Normalizer(size_g, world_size, clip=args.goal_clip, idx=goal_norm_idx)
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
        if dist and world_size > 1:
            self.init_dist()
        if self.args.save:
            self.PATH = Path(__file__).parents[2] / "saves" / self.args.env
            if not self.PATH.is_dir():
                self.PATH.mkdir(parents=True, exist_ok=True)
            # Create a unique backup path with the current time and resolve name collisions
            if self.rank == 0:
                self.BACKUP_PATH = self.PATH / "backup" / datetime.now().strftime("%Y_%m_%d_%H_%M")
                if not self.BACKUP_PATH.is_dir():
                    self.BACKUP_PATH.mkdir(parents=True, exist_ok=True)
                else:
                    t = 1
                    while self.BACKUP_PATH.is_dir():
                        self.BACKUP_PATH = self.PATH / "backup" / (
                            datetime.now().strftime("%Y_%m_%d_%H_%M") + f"_({t})")
                        t += 1
                    self.BACKUP_PATH.mkdir(parents=True, exist_ok=True)
            else:
                self.BACKUP_PATH = None

    def train(self):
        """Train a policy to solve the environment with DDPG.

        Trajectories are resampled with HER to solve sparse reward environments. Supports
        distributed training across multiple processes.

        `DDPG paper <https://arxiv.org/pdf/1509.02971.pdf>`_

        `HER paper <https://arxiv.org/pdf/1707.01495.pdf>`_
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
                    self.actor.noise_process.reset()
                self._update_norm(ep_buffer)
                for _ in range(self.args.batches):
                    self._train_agent()
                self.actor.update_target(self.args.tau)
                self.critic.update_target(self.args.tau)
            epoch_end = time.perf_counter()
            av_success = self.eval_agent()
            if hasattr(self.env, "epoch_callback"):
                assert callable(self.env.epoch_callback)
                self.env.epoch_callback(epoch, av_success)
            if self.rank == 0:
                ep_success.append(av_success)
                ep_time.append(epoch_end - epoch_start)
                success_log.set_description_str("Current success rate: {:.3f}".format(av_success))
                status_bar.update()
                if self.args.save:
                    self.save_models()
                    self.save_plots(ep_success, ep_time)
                    self.save_stats(ep_success, ep_time)
            if av_success >= self.args.early_stop and self.env.early_stop_ok:
                if self.rank == 0 and self.args.save:
                    self.save_models()
                    self.save_models(path=self.BACKUP_PATH)
                    self.save_plots(ep_success, ep_time)
                    self.save_stats(ep_success, ep_time)
                return
        if self.rank == 0 and self.args.save:
            self.save_models()
            self.save_models(path=self.BACKUP_PATH)
            self.save_plots(ep_success, ep_time)
            self.save_stats(ep_success, ep_time)

    def _train_agent(self):
        """Train the agent and critic network with experience sampled from the replay buffer."""
        states, actions, rewards, next_states, goals = self.buffer.sample(self.args.batch_size)
        obs_T, obs_next_T = self.wrap_obs(states, goals), self.wrap_obs(next_states, goals)
        actions_T = torch.as_tensor(actions, dtype=torch.float32)
        rewards_T = torch.as_tensor(rewards, dtype=torch.float32)
        with torch.no_grad():
            next_actions_T = self.actor.target(obs_next_T)
            next_q_T = self.critic.target(obs_next_T, next_actions_T)
            rewards_T = rewards_T + self.args.gamma * next_q_T  # No dones in fixed length episode
            # Clip to minimum reward possible, geometric sum from 0 to inf with gamma and -1 rewards
            torch.clip(rewards_T, -1 / (1 - self.args.gamma), 0, out=rewards_T)
        q_T = self.critic(obs_T, actions_T)
        critic_loss_T = (rewards_T - q_T).pow(2).mean()

        # Regularize tanh activation in PosePolicyNets to prevent saturation and possibly parallel
        # rotation vectors (leads to random orientations and bad policies, see ``PosePolicyNet`` for
        # details)
        if isinstance(self.actor.action_net, PosePolicyNet):
            actions_T, activation_T = self.actor.action_net.forward_include_network_output(obs_T)
            activation_norm_T = self.args.action_norm * (activation_T[..., 3:9]).pow(2).mean()
        else:
            actions_T = self.actor(obs_T)
        next_q_T = self.critic(obs_T, actions_T)
        if isinstance(self.actor.action_net, PosePolicyNet):
            actor_loss_T = -next_q_T.mean() + activation_norm_T
        else:
            actor_loss_T = -next_q_T.mean()
        self.actor.backward_step(actor_loss_T)
        self.critic.backward_step(critic_loss_T)

    def _update_norm(self, ep_buffer: TrajectoryBuffer):
        """Update the normalizers with an episode of play experience.

        Samples the trajectory instead of taking every experience to create a goal distribution that
        is equal to what the networks encouter.

        Args:
            ep_buffer: Buffer containing a trajectory of replay experience.
        """
        states, goals = ep_buffer.buffer["s"], ep_buffer.buffer["g"]
        self.state_norm.update(states)
        self.goal_norm.update(goals)

    def eval_agent(self) -> float:
        """Evaluate the current agent performance on the gym task.

        Runs `args.evals` times and averages the success rate. If distributed training is enabled,
        further averages over all distributed evaluations.
        """
        self.actor.eval()
        success = 0
        self.env.use_info(True)
        for _ in range(self.args.evals):
            state, goal, _ = unwrap_obs(self.env.reset())
            for t in range(self.T):
                with torch.no_grad():
                    action = self.actor.select_action(self.wrap_obs(state, goal))
                next_obs, _, _, info = self.env.step(action)
                state, goal, _ = unwrap_obs(next_obs)
            success += info["is_success"]
        self.actor.train()
        self.env.use_info(False)
        if self.dist:
            success_rate = np.array([success / self.args.evals])
            MPI.COMM_WORLD.Allreduce(success_rate, success_rate, op=MPI.SUM)
            return success_rate[0] / self.world_size
        return success / self.args.evals

    def save_models(self, path: Optional[Path] = None):
        """Save the actor and critic network and the normalizers for testing and inference.

        Saves are located under `/save/<env_name>/` by default. Only saves if called from rank 0,
        returns without saving otherwise.

        Args:
            path: Path to the save directory.
        """
        if self.rank != 0:
            logger.warning(f"save_models called with rank {self.rank}. Exiting without save.")
            return
        path = path or self.PATH
        torch.save(self.actor.action_net.state_dict(), path / "actor.pt")
        torch.save(self.critic.critic_net.state_dict(), path / "critic.pt")
        self.state_norm.save(path / "state_norm.pkl")
        self.goal_norm.save(path / "goal_norm.pkl")

    def save_plots(self, ep_success: List[float], ep_time: List[float]):
        """Generate and save a plot from training statistics.

        Saves are located under `/save/<env_name>/`. Additional backup saves with the current
        timestamp are created under `/save/<env_name>/.backup/<date>`. Can only be called from rank
        0.

        Args:
            ep_success: Episode agent evaluation success rate.
            ep_time: Episode training process time. Excludes the evaluation timing.
        """
        assert self.rank == 0
        fig, ax = plt.subplots(1, 2, figsize=(15, 4))
        ax[0].plot(ep_success)
        ax[0].set_xlabel('Episode')
        ax[0].set_ylabel('Success rate')
        ax[0].set_ylim([0., 1.])
        ax[0].set_title('Agent performance over time')
        ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))

        ax[1].plot(ep_time)
        ax[1].set_xlabel('Episode')
        ax[1].set_ylabel('Time in s')
        ax[1].set_title('Episode compute times')
        ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(self.PATH / "stats.png")
        plt.savefig(self.BACKUP_PATH / "stats.png")
        plt.close()

    def save_stats(self, ep_success: List[float], ep_time: List[float]):
        """Save the current stats to a json file.

        Saves are located under `/save/<env_name>/`. Additional backup saves with the current
        timestamp are created under `/save/<env_name>/.backup/<date>`. Can only be called from rank
        0.

        Args:
            ep_success: Episode evaluation success rate array.
            ep_time: Episode compute time array.
        """
        world_size = None if self.world_size == 1 else self.world_size
        stats = {
            "ep_success": ep_success,
            "ep_time": ep_time,
            "args": vars(self.args),
            "mpi": world_size
        }
        assert self.rank == 0
        with open(self.PATH / "stats.json", "w") as f:
            json.dump(stats, f)
        with open(self.BACKUP_PATH / "stats.json", "w") as f:
            json.dump(stats, f)

    def load_pretrained(self, path: Path):
        """Load pretrained networks for the actor, critic and normalizers."""
        if not path.is_dir():
            raise NotADirectoryError("Path must point to a valid directory.")
        self.actor.load(path / "actor.pt")
        self.critic.load(path / "critic.pt")
        self.state_norm.load(path / "state_norm.pkl")
        self.goal_norm.load(path / "goal_norm.pkl")

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

    def init_dist(self):
        """Configure actor, critic and normalizers for distributed training."""
        self.actor.init_dist()
        self.critic.init_dist()
        self.state_norm.init_dist()
        self.goal_norm.init_dist()
        self.dist = True
