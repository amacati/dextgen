"""``DDPG`` module encapsulating the Deep Deterministic Policy Gradient algorithm.

:class:`.DDPG` initializes the actor, critic, normalizers and noise processes, manages the
synchronization between MPI nodes and takes care of checkpoints during training as well as network
loading if starting from pre-trained networks. It assumes dictionary gym environments.
"""

import argparse
import logging
from typing import Optional
from pathlib import Path
import time

import numpy as np
import torch
from tqdm import tqdm
import gym
from mpi4py import MPI

from mp_rl.core.utils import unwrap_obs
from mp_rl.core.noise import UniformNoise
from mp_rl.core.actor import Actor, PosePolicyNet
from mp_rl.core.critic import Critic
from mp_rl.core.normalizer import Normalizer
from mp_rl.core.replay_buffer import HERBuffer, TrajectoryBuffer
from mp_rl.utils import Logger

logger = logging.getLogger(__name__)


class DDPG:
    """Deep Deterministic Policy Gradient algorithm class.

    Uses a state/goal normalizer and the HER sampling method to solve sparse reward environments.
    """

    def __init__(self,
                 env: gym.Env,
                 args: argparse.Namespace,
                 logger: Logger,
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
        self.args = args
        size_s = len(env.observation_space["observation"].low)
        size_a = len(env.action_space.low)
        size_g = len(env.observation_space["desired_goal"].low)
        noise_process = UniformNoise(size_a)
        # Create actor and critic networks
        self.actor = Actor(args.actor_net_type, size_s + size_g, size_a, args.actor_net_nlayers,
                           args.actor_net_layer_width, noise_process, args.actor_lr, args.eps,
                           args.action_clip, args.grad_clip)
        self.critic = Critic(size_s + size_g, size_a, args.critic_net_nlayers,
                             args.critic_net_layer_width, args.critic_lr, args.grad_clip)
        # Create normalizers
        state_norm_idx = getattr(args, "state_norm_idx", None)
        goal_norm_idx = getattr(args, "goal_norm_idx", None)
        self.state_norm = Normalizer(size_s, world_size, clip=args.state_clip, idx=state_norm_idx)
        self.goal_norm = Normalizer(size_g, world_size, clip=args.goal_clip, idx=goal_norm_idx)
        # Create replay buffer
        self.T = env._max_episode_steps
        self.buffer = HERBuffer(size_s,
                                size_a,
                                size_g,
                                self.T,
                                args.her_n_sampled_goal,
                                args.buffer_size,
                                reward_fun=self.env.compute_reward)
        self.action_max = torch.as_tensor(self.env.action_space.high, dtype=torch.float32)
        # Initialize distributed training
        self.dist = False
        self.world_size = world_size
        self.rank = rank
        self._total_train_steps = self.args.n_total_steps // self.T // self.world_size
        self._last_eval_steps = 0
        if dist and world_size > 1:
            self.init_dist()
        self.logger = logger

    def train(self):
        """Train a policy to solve the environment with DDPG.

        Trajectories are resampled with HER to solve sparse reward environments. Supports
        distributed training across multiple processes.

        `DDPG paper <https://arxiv.org/pdf/1509.02971.pdf>`_

        `HER paper <https://arxiv.org/pdf/1707.01495.pdf>`_
        """
        total_train_steps = self.args.n_total_steps // self.T // self.world_size
        if self.rank == 0:
            status_bar = tqdm(total=total_train_steps,
                              desc="Training steps",
                              position=0,
                              leave=True,
                              dynamic_ncols=True)
            success_log = tqdm(total=0,
                               position=1,
                               bar_format='{desc}',
                               leave=True,
                               dynamic_ncols=True)
        current_step = 0
        training_start = time.time()
        # Main training loop
        for epoch in range(total_train_steps):
            for _ in range(self.args.rollouts):
                ep_buffer = self.buffer.get_trajectory_buffer()
                obs = self.env.reset()
                state, goal, agoal = unwrap_obs(obs)
                for _ in range(self.T):
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
            current_step += self.world_size * self.T * self.args.rollouts
            # Perform policy and critic network updates
            for train_step in range(self.args.gradient_steps):
                self._train_agent(log=train_step == self.args.gradient_steps - 1)
            self.actor.update_target(self.args.tau)
            self.critic.update_target(self.args.tau)
            # Evaluate the current performance of the agent
            if current_step - self._last_eval_steps >= self.args.eval_interval:
                self._last_eval_steps = current_step
                av_success, av_reward = self.eval_agent()
                log = {
                    "eval/success_rate": av_success,
                    "eval/mean_reward": av_reward,
                    "time/time_elapsed": time.time() - training_start,
                    "time/total_timesteps": current_step,
                    "time/fps": current_step / (time.time() - training_start)
                }
                self.logger.log(log, current_step)
                if self.rank == 0:
                    success_log.set_description_str(f"Success rate: {av_success:.2f}")
                    if self.args.save:
                        self.save_models(self.logger.path)
                if av_success >= self.args.early_stop and self.env.early_stop_ok:
                    break
            if self.rank == 0:
                status_bar.update()
            if hasattr(self.env, "epoch_callback"):
                assert callable(self.env.epoch_callback)
                self.env.epoch_callback(epoch, av_success)
        if self.rank == 0 and self.args.save:
            self.save_models(self.logger.path)

    def _train_agent(self, log: bool = False):
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
        if log:
            self.logger.log({
                "train/actor_loss": actor_loss_T.item(),
                "train/critic_loss": critic_loss_T.item()
            })

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

    def eval_agent(self) -> tuple[float, float]:
        """Evaluate the current agent performance on the gym task.

        Runs `args.num_evals` times and averages the success rate. If distributed training is
        enabled, further averages over all distributed evaluations.
        """
        self.actor.eval()
        success = 0
        self.env.use_info(True)
        total_reward = 0
        for _ in range(self.args.num_evals):
            state, goal, _ = unwrap_obs(self.env.reset())
            for t in range(self.T):
                with torch.no_grad():
                    action = self.actor.select_action(self.wrap_obs(state, goal))
                next_obs, reward, _, info = self.env.step(action)
                total_reward += reward
                state, goal, _ = unwrap_obs(next_obs)
            success += info["is_success"]
        self.actor.train()
        self.env.use_info(False)
        if self.dist:
            eval_info = np.array([success, total_reward]) / self.args.num_evals
            world_eval_info = np.zeros_like(eval_info)
            MPI.COMM_WORLD.Allreduce(eval_info, world_eval_info, op=MPI.SUM)
            return world_eval_info[0] / self.world_size, world_eval_info[1] / self.world_size
        return success / self.args.num_evals, total_reward / self.args.num_evals

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
