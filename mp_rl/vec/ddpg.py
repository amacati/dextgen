"""``DDPG`` module encapsulating the Deep Deterministic Policy Gradient algorithm.

:class:`.DDPG` initializes the actor, critic, normalizers and noise processes, manages the
synchronization between MPI nodes and takes care of checkpoints during training as well as network
loading if starting from pre-trained networks. It assumes dictionary gymnasium environments.
"""

import argparse
import logging
from typing import Optional
from pathlib import Path
import time

import numpy as np
import torch
from tqdm import tqdm
import gymnasium
import einops

from mp_rl.vec.utils import unwrap_obs
from mp_rl.vec.noise import UniformNoise
from mp_rl.vec.actor import Actor
from mp_rl.vec.critic import Critic
from mp_rl.vec.normalizer import Normalizer
from mp_rl.vec.replay_buffer import HERBuffer, TrajectoryBuffer
from mp_rl.utils import Logger

logger = logging.getLogger(__name__)


def default_reward_fn(achieved_goal: np.ndarray, goal: np.ndarray, _) -> np.ndarray:
    """Compute the agent reward for the achieved goal.

    Args:
        achieved_goal: Achieved goal.
        goal: Desired goal.
        _: Ignored argument.

    Returns:
        The default reward for the given state and goal.
    """
    TARGET_THRESHOLD = 0.05
    achieved_goal = achieved_goal[..., :3]
    assert achieved_goal.shape == goal.shape
    return -(np.linalg.norm(achieved_goal - goal, axis=-1) > TARGET_THRESHOLD).astype(np.float32)


class DDPG:
    """Deep Deterministic Policy Gradient algorithm class.

    Uses a state/goal normalizer and the HER sampling method to solve sparse reward environments.
    """

    def __init__(self,
                 env: gymnasium.vector.VectorEnv,
                 eval_env: gymnasium.vector.VectorEnv,
                 args: argparse.Namespace,
                 logger: Logger,
                 seed: int = None):
        """Initialize the Actor, Critic, HERBuffer and activate MPI synchronization if required.

        Args:
            env: OpenAI dictionary gymnasium environment.
            eval_env: OpenAI dictionary gymnasium environment for evaluation.
            args: User settings and configs merged into a single namespace.
        """
        self.env = env
        self.eval_env = eval_env
        self.seed = seed
        # Rewards are calculated from HER buffer. Disable computation for runtime improvement
        self.args = args
        size_s = len(env.single_observation_space["observation"].low)
        size_a = len(env.single_action_space.low)
        size_g = len(env.single_observation_space["desired_goal"].low)
        noise_process = UniformNoise(env.num_envs, size_a)
        # Create actor and critic networks
        self.actor = Actor(size_s=size_s + size_g,
                           size_a=size_a,
                           nlayers=args.actor_net_nlayers,
                           layer_width=args.actor_net_layer_width,
                           noise_process=noise_process,
                           lr=args.actor_lr,
                           eps=args.eps,
                           action_clip=args.action_clip,
                           grad_clip=args.grad_clip)
        self.critic = Critic(size_s=size_s + size_g,
                             size_a=size_a,
                             nlayers=args.critic_net_nlayers,
                             layer_width=args.critic_net_layer_width,
                             lr=args.critic_lr,
                             grad_clip=args.grad_clip)
        # Create normalizers
        state_norm_idx = getattr(args, "state_norm_idx", None)
        goal_norm_idx = getattr(args, "goal_norm_idx", None)
        self.state_norm = Normalizer(size_s, clip=args.state_clip, idx=state_norm_idx)
        self.goal_norm = Normalizer(size_g, clip=args.goal_clip, idx=goal_norm_idx)
        # Create replay buffer
        reward_fn = getattr(env.unwrapped, "compute_reward", None) or default_reward_fn
        self.N = env.num_envs
        self.T = env.spec.max_episode_steps
        self.buffer = HERBuffer(size_s=size_s,
                                size_a=size_a,
                                size_g=size_g,
                                N=self.N,
                                T=self.T,
                                k=args.her_n_sampled_goal,
                                max_samples=args.buffer_size,
                                reward_fun=reward_fn)
        self._last_eval_steps = 0
        self.logger = logger

    def train(self):
        """Train a policy to solve the environment with DDPG.

        Trajectories are resampled with HER to solve sparse reward environments.

        `DDPG paper <https://arxiv.org/pdf/1509.02971.pdf>`_

        `HER paper <https://arxiv.org/pdf/1707.01495.pdf>`_
        """
        epochs = self.args.n_total_steps // self.env.num_envs // self.T
        status_bar = tqdm(total=epochs,
                          desc="Training steps",
                          position=0,
                          leave=True,
                          dynamic_ncols=True)
        success_log = tqdm(total=0, position=1, bar_format='{desc}', leave=True, dynamic_ncols=True)
        current_step = 0
        training_start = time.time()
        assert self.env.num_envs == 32, "Only 32 environments are supported for now."
        # Main training loop
        for epoch in range(epochs):
            ep_buffer = self.buffer.get_trajectory_buffer()
            seed = self.seed + epoch * self.env.num_envs if isinstance(self.seed, int) else None
            obs, _ = self.env.reset(seed=seed)
            state, goal, agoal = unwrap_obs(obs)
            for _ in range(self.T):
                with torch.no_grad():
                    action = self.actor.select_action(self.wrap_obs(state, goal))
                next_obs, _, _, _, info = self.env.step(action)
                next_state, _, next_agoal = unwrap_obs(next_obs)
                ep_buffer.append(state, action, goal, agoal)
                state, agoal = next_state, next_agoal
            # Vector environments automatically reset after T steps. The last observation is
            # already the first observation of the next episode, so we have to get the final
            # observation from the info dictionary.
            final_obs = info["final_observation"]
            state = np.array([o["observation"] for o in final_obs])
            agoal = np.array([o["achieved_goal"] for o in final_obs])
            ep_buffer.append(state, agoal)
            self.buffer.append(ep_buffer)
            self.actor.noise_process.reset()
            self._update_norm(ep_buffer)
            current_step += self.T * self.env.num_envs
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
                success_log.set_description_str(f"Success rate: {av_success:.2f}")
                if self.args.save:
                    self.save_models(self.logger.path)
                if av_success >= self.args.early_stop:
                    break
            status_bar.update()
            if hasattr(self.env.unwrapped, "epoch_callback"):
                assert callable(self.env.unwrapped.epoch_callback)
                self.env.epoch_callback(epoch, av_success)
        if self.args.save:
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
            assert rewards_T.shape == next_q_T.shape
        q_T = self.critic(obs_T, actions_T)
        assert rewards_T.shape == q_T.shape
        critic_loss_T = (rewards_T - q_T).pow(2).mean()
        actions_T = self.actor(obs_T)
        next_q_T = self.critic(obs_T, actions_T)
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
        states = einops.rearrange(ep_buffer.buffer["s"], "n t d -> (n t) d")
        goals = einops.rearrange(ep_buffer.buffer["g"], "n t d -> (n t) d")
        self.state_norm.update(states)
        self.goal_norm.update(goals)

    def eval_agent(self) -> tuple[float, float]:
        """Evaluate the current agent performance on the gymnasium task.

        Runs `args.num_evals` times and averages the success rate.
        """
        self.actor.eval()
        success = 0
        total_reward = 0
        num_evals = self.args.num_evals // self.eval_env.num_envs
        seed = self._last_eval_steps + self.seed if isinstance(self.seed, int) else None
        for _ in range(num_evals):
            state, goal, _ = unwrap_obs(self.eval_env.reset(seed=seed)[0])
            for t in range(self.T):
                with torch.no_grad():
                    action = self.actor.select_action(self.wrap_obs(state, goal))
                next_obs, reward, _, _, _ = self.eval_env.step(action)
                total_reward += np.sum(reward)
                state, goal, _ = unwrap_obs(next_obs)
            success += np.sum(reward >= 0)
        self.actor.train()
        av_success = success / (num_evals * self.eval_env.num_envs)
        av_reward = total_reward / (num_evals * self.eval_env.num_envs)
        return av_success, av_reward

    def save_models(self, path: Optional[Path] = None):
        """Save the actor and critic network and the normalizers for testing and inference.

        Saves are located under `/save/<env_name>/` by default.

        Args:
            path: Path to the save directory.
        """
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
