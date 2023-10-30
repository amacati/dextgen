from __future__ import annotations
from typing import Callable
from abc import abstractmethod, ABC

import numpy as np


class ReplayBuffer(ABC):
    """Abstract base class for replay buffers."""

    def __init__(self):
        """Abstract init method."""
        super().__init__()

    @abstractmethod
    def sample(self, *_):
        """Abstract sample method."""

    @abstractmethod
    def append(self, *_):
        """Abstract append method."""

    @abstractmethod
    def __len__(self):
        """Abstract length method."""


class HERBuffer(ReplayBuffer):
    """Hindsight Experience Replay Buffer class.

    Buffer expects episodes of fixed time horizon T.
    """

    def __init__(self, size_env: int, size_s: int, size_a: int, size_g: int, T: int, k: int,
                 max_samples: int, reward_fn: Callable):
        """Initialize the buffers for states, actions, goals and achieved goals.

        Args:
            size_env: Number of environments.
            size_s: State dimension.
            size_a: Action dimension.
            size_g: Goal dimension.
            T: Trajectory length in time steps.
            k: HER resampling factor.
            max_samples: Maximum buffer size in single experience (= transition) samples.
            reward_fn: Environment reward function.
        """
        self.size_traj = max_samples // T
        self.size_env = size_env
        self.size_s = size_s
        self.size_a = size_a
        self.size_g = size_g
        self._step_idx = 0
        self._traj_idx = 0
        self._len = 0
        self.T = T  # Episodes have a fixed time horizon T
        # observation and achieved_goal have one more time step than action and desired_goal because
        # we do not store the next observation explicity. Instead, we use the next observation in
        # the trajectory to eliminate redundancy and save memory. This requires us to store one more
        # observation in the observation buffer (next_observation = observation[t + 1]). Hence, the
        # buffer is T + 1 time steps long.
        self.buffer = {
            "observation": np.zeros([self.size_traj, T + 1, size_s], dtype=np.float32),
            "action": np.zeros([self.size_traj, T, size_a], dtype=np.float32),
            "desired_goal": np.zeros([self.size_traj, T, size_g], dtype=np.float32),
            "achieved_goal": np.zeros([self.size_traj, T + 1, size_g], dtype=np.float32),
            "_valid": np.zeros(self.size_traj, dtype=bool),
        }
        self.k = k
        self.p_her = 1 - 1. / (1 + k)
        self.reward_fn = reward_fn

    def append(self, sample: dict):
        """Append experience in a trajectory buffer to the replay buffer.

        Args:
            sample: A dictionary of transitions from a vectorized environment
        """
        assert sample["observation"].shape[0] == self.size_env
        idx_start, idx_end = self._traj_idx, (self._traj_idx + self.size_env) % self.size_traj
        chunk = self.size_env if idx_start < idx_end else self.size_traj - idx_start
        idx_chunk = idx_start + chunk if idx_start < idx_end else 0
        # We mark all trajectories that are currently being overwritten as invalid to prevent the
        # sampling of inconsistent trajectories.
        if self._step_idx == 0:
            self.buffer["_valid"][idx_start:idx_start + chunk] = False
            self.buffer["_valid"][idx_chunk:idx_end] = False
        for key in sample.keys():
            # We chunk the sample. If the trajectory buffer does not overflow, the chunk size is of
            # size self.size_env and all assignments are done in one go. If the trajectory buffer
            # overflows, we need to split the chunk into two parts. The first part is of size
            # `chunk` and is assigned to the remaining space in the buffer. The second part is of
            # size `self.size_env` - `chunk` and overwrites the oldest samples at the beginning of
            # the buffer.
            self.buffer[key][idx_start:idx_start + chunk, self._step_idx] = sample[key][:chunk]
            self.buffer[key][idx_chunk:idx_end, self._step_idx] = sample[key][chunk:]
        self._step_idx = (self._step_idx + 1) % (self.T + 1)
        # Finished trajectories are marked as valid again.
        if self._step_idx == 0:
            self.buffer["_valid"][idx_start:idx_start + chunk] = True
            self.buffer["_valid"][idx_chunk:idx_end] = True
            self._traj_idx = (self._traj_idx + self.size_env) % self.size_traj
        self._len = min(self._len + self.size_env, self.size_traj * self.T)

    def __len__(self):
        return self._len

    def sample(self, batch_size: int) -> tuple[np.ndarray]:
        valid_samples = np.argwhere(self.buffer["_valid"]).flatten()
        assert len(valid_samples) * self.T >= batch_size, "Less samples than required in the buffer"
        t_idx = np.random.choice(valid_samples, batch_size, replace=True)
        s_idx = np.random.randint(0, self.T, batch_size)
        # Copy transitions to avoid changing the ground truth values after sampling
        goals = self.buffer["desired_goal"][t_idx, s_idx].copy()
        h_idx = np.where(np.random.uniform(size=batch_size) < self.p_her)
        t_offset = (np.random.uniform(size=batch_size) * (self.T - s_idx)).astype(int)
        f_idx = (s_idx + 1 + t_offset)[h_idx]
        goals[h_idx] = self.buffer["achieved_goal"][t_idx[h_idx], f_idx].copy()
        rewards = self.reward_fn(self.buffer["achieved_goal"][t_idx, s_idx + 1], goals, None)
        rewards = np.expand_dims(rewards, 1)
        next_states = self.buffer["observation"][t_idx, s_idx + 1].copy()
        states = self.buffer["observation"][t_idx, s_idx].copy()
        actions = self.buffer["action"][t_idx, s_idx].copy()
        return states, actions, rewards, next_states, goals
