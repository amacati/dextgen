"""The ``replay_buffer`` module contains various buffer types.

Buffers follow a common interface defined in :class:`.ReplayBuffer`. Actual implementations contain
the :class:`.HERBuffer` as well as a default :class:`.MemoryBuffer` without special sampling.

The :class:`.TrajectoryBuffer` is not a replay buffer, but holds experience from an episode of fixed
time length. It is a convenience class for storing experience in an orderly fashion during training.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Union, Tuple, List
from collections import deque

import numpy as np

from mp_rl.utils import import_guard

if import_guard():
    from collections.abc import Iterator, KeysView, ValuesView, ItemsView  # noqa: TC003, is guarded


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


class TrajectoryBuffer:
    """Trajectory buffer to hold a single trajectory of fixed time horizon from a single episode."""

    def __init__(self, size_s: int, size_a: int, size_g: int, T: int):
        """Initialize the buffers for states, actions, goals and achieved goals.

        Args:
            size_s: State dimension.
            size_a: Action dimension.
            size_g: Goal dimension.
            T: Trajectory length in time steps.
        """
        self.T = T
        # Both state and achieved goal hold one additional sample for the last simulation step
        self.buffer = {
            "s": np.zeros([T + 1, size_s]),
            "a": np.zeros([T, size_a]),
            "g": np.zeros([T, size_g]),
            "ag": np.zeros([T + 1, size_g])
        }
        self.t = 0

    def append(self, *args: Tuple[np.ndarray]):
        """Append experience to the buffer.

        If the buffer is not full, args is a tuple of 4 arrays: `state`, `action`, `goal`,
        `achieved goal`. If the buffer is full, one additional observation tuple of 2 arrays `state`
        and `achieved goal` should be appended.

        Args:
            args: Experience samples. Either state, action, goal, achieved goal arrays, or only
                state and achieved goal array.

        Raises:
            AssertionError: Trajectory append arguments don't match the current state of the buffer.
            IndexError: Append call after buffer has been filled.
        """
        if self.t < self.T:
            assert len(args) == 4, "Trajectory appends require 4 arguments"
            for buffer, arr in zip(self.buffer.values(), args):
                buffer[self.t] = arr.copy()
            self.t += 1
        elif self.t == self.T:
            assert len(args) == 2, "Final trajectory append requires 2 arguments"
            self.buffer["s"][self.T] = args[0].copy()
            self.buffer["ag"][self.T] = args[1].copy()
        elif self.t > self.T:
            raise IndexError("Tried to add more samples to trajectory than its length!")

    def __getitem__(self, key: str) -> np.ndarray:
        """Expose the getitem method of the internal buffer.

        Args:
            key: The buffer dict key. Must be 's', 'a', 'g' or 'ag'.

        Returns:
            The buffer array for the key.
        """
        return self.buffer[key]

    def __len__(self) -> int:
        """Get the current length of the buffer.

        Note:
            The buffer size itself is static. The length refers to the number of timesteps that are
            filled with transitions.

        Returns:
            The current buffer length.
        """
        return self.t

    def clear(self):
        """Clear the buffer arrays."""
        self.buffer = {key: np.empty((self.T, size_b)) for key, size_b in self.buffer_sizes.items()}
        self.t = 0

    def keys(self) -> KeysView:
        """Expose the key method of the internal buffer.

        Returns:
            A key view of the buffer
        """
        return self.buffer.keys()

    def values(self) -> ValuesView:
        """Expose the values method of the internal buffer.

        Returns:
            A values view of the buffer
        """
        return self.buffer.values()

    def items(self) -> ItemsView:
        """Expose the items method of the internal buffer.

        Returns:
            An items view of the buffer
        """
        return self.buffer.items()

    def __iter__(self) -> Iterator:
        """Expose the __iter__ method of the internal buffer.

        Returns:
            The buffer dict iterator.
        """
        return self.buffer.__iter__(self)


class HERBuffer(ReplayBuffer):
    """Hindsight Experience Replay Buffer class.

    Buffer expects episodes of fixed time horizon T.
    """

    def __init__(self,
                 size_s: int,
                 size_a: int,
                 size_g: int,
                 T: int,
                 k: int,
                 max_samples: int,
                 reward_fun: Callable,
                 sample_mode: str = "her"):
        """Initialize the buffers for states, actions, goals and achieved goals.

        Args:
            size_s: State dimension.
            size_a: Action dimension.
            size_g: Goal dimension.
            T: Trajectory length in time steps.
            k: HER resampling factor.
            max_samples: Maximum buffer size in single experience (= transition) samples.
            reward_fun: Environment reward function.
            sample_mode: Sample generation mode. Currently supports `her` and `default`.

        Raises:
            AssertionError: Sample mode is not supported.
        """
        self.size = max_samples // T
        self.size_s = size_s
        self.size_a = size_a
        self.size_g = size_g
        self.curr_size = 0
        self.T = T  # Episodes have a fixed time horizon T
        # Keys: state, action, goal, achieved goal
        self.buffer = {
            "s": np.zeros([self.size, T + 1, size_s]),
            "a": np.zeros([self.size, T, size_a]),
            "g": np.zeros([self.size, T, size_g]),
            "ag": np.zeros([self.size, T + 1, size_g])
        }
        self.k = k
        self.p_her = 1 - 1. / (1 + k)
        assert sample_mode in ["her", "default"]
        self.sample_mode = sample_mode
        self.reward_fun = reward_fun or self._reward_fun

    def append(self, ep_buffer: TrajectoryBuffer):
        """Append experience in a trajectory buffer to the replay buffer.

        Args:
            ep_buffer: Trajectory buffer filled with replay experience.
        """
        self._validate_episode(ep_buffer)
        idx = self.curr_size if self.curr_size < self.size else np.random.randint(0, self.size)
        for key, val in ep_buffer.items():  # Keys are already validated to match the buffer
            self.buffer[key][idx] = val
        self.curr_size = min(self.curr_size + 1, self.size)

    def _validate_episode(self, episode: TrajectoryBuffer):
        if len(episode) != self.T:
            raise RuntimeError("Episode length has to match time horizon T")

    def sample(self, batch_size: int) -> Tuple[np.ndarray]:
        """Sample a batch from the replay buffer.

        Args:
            batch_size: Batch size

        Returns:
            The sampled batch.

        Raises:
            RuntimeError: Sample mode is not supported.
        """
        if batch_size > self.size * self.T:
            raise RuntimeError("Batch size batch_size exceeds buffer contents")
        if self.sample_mode == "default":
            return default_sampling(self.buffer["s"][:self.curr_size],
                                    self.buffer["a"][:self.curr_size],
                                    self.buffer["g"][:self.curr_size],
                                    self.buffer["ag"][:self.curr_size], batch_size, self.reward_fun)
        elif self.sample_mode == "her":
            return her_sampling(self.buffer["s"][:self.curr_size],
                                self.buffer["a"][:self.curr_size],
                                self.buffer["g"][:self.curr_size],
                                self.buffer["ag"][:self.curr_size], batch_size, self.p_her,
                                self.reward_fun)
        raise RuntimeError("Unsupported sample mode!")

    def _reward_fun(self, *_):
        raise NotImplementedError("Reward function has to be specified by user before use")

    def __len__(self) -> int:
        """Get the current length of the buffer.

        Note:
            The buffer size itself is static. The length refers to the number of entries that are
            filled with trajectories.

        Returns:
            The current size.
        """
        return self.curr_size

    def get_trajectory_buffer(self) -> TrajectoryBuffer:
        """Create a trajectory buffer of appropriate dimensions for this HERBuffer.

        Returns:
            The trajectory buffer.
        """
        return TrajectoryBuffer(self.size_s, self.size_a, self.size_g, self.T)


def her_sampling(states: np.ndarray, actions: np.ndarray, goals: np.ndarray, agoals: np.ndarray,
                 batch_size: int, p_her: float, reward_fun: Callable) -> Tuple[np.ndarray]:
    """Sample a batch at random with HER goal resampling from replay experience.

    Args:
        states: State batch array.
        actions: Action batch array.
        goals: Goal batch array.
        agoals: Achieved goals batch array.
        batch_size: Batch size.
        p_her: Probability of resampling a goal with an achieved goal.
        reward_fun: Reward function of the environment to recalculate the rewards.

    Returns:
        A tuple of the sampled state, action, reward, next_state, goal batch.

    Raises:
        Assertion error: Dimension check on states failed.
    """
    assert states.ndim == 3, "Requires tensors of dimensions (episode, timestep, data_dim)"
    neps, T, _ = actions.shape
    e_idx = np.random.randint(0, neps, batch_size)
    t_idx = np.random.randint(0, T, batch_size)
    # Copy transitions to avoid changing the ground truth values after sampling
    goals = goals[e_idx, t_idx].copy()
    h_idx = np.where(np.random.uniform(size=batch_size) < p_her)
    t_offset = (np.random.uniform(size=batch_size) * (T - t_idx)).astype(int)
    f_idx = (t_idx + 1 + t_offset)[h_idx]
    goals[h_idx] = agoals[e_idx[h_idx], f_idx].copy()
    rewards = reward_fun(agoals[e_idx, t_idx + 1], goals, None)
    rewards = np.expand_dims(rewards, 1)
    next_states = states[e_idx, t_idx + 1].copy()
    states, actions = states[e_idx, t_idx].copy(), actions[e_idx, t_idx].copy()
    return states, actions, rewards, next_states, goals


def default_sampling(states: np.ndarray, actions: np.ndarray, goals: np.ndarray, agoals: np.ndarray,
                     batch_size: int, reward_fun: Callable) -> Tuple[np.ndarray]:
    """Sample a batch at random without goal changes from replay experience.

    Args:
        states: State batch array.
        actions: Action batch array.
        goals: Goal batch array.
        agoals: Achieved goals batch array.
        batch_size: Batch size.
        reward_fun: Reward function of the environment to recalculate the rewards.

    Returns:
        A tuple of the sampled state, action, reward, next_state, goal batch.

    Raises:
        Assertion error: Dimension check on states failed.
    """
    assert states.ndim == 3, "Requires tensors of dimensions (episode, timestep, data_dim)"
    neps, T, _ = actions.shape
    e_idx = np.random.randint(0, neps, batch_size)
    t_idx = np.random.randint(0, T, batch_size)
    next_states, goals = states[e_idx, t_idx + 1].copy(), goals[e_idx, t_idx].copy()
    states, actions = states[e_idx, t_idx].copy(), actions[e_idx, t_idx].copy()
    rewards = np.expand_dims(reward_fun(agoals[e_idx, t_idx + 1], goals, None), 1)
    return states, actions, rewards, next_states, goals


class MemoryBuffer(ReplayBuffer):
    """A simple ReplayBuffer that stores single replay experience samples."""

    def __init__(self, maxlen: int = 10000):
        """Initialize the buffer.

        Args:
            maxlen: Maximum buffer size.
        """
        # Use of deque makes sure appends drop old experience samples after reaching the size limit
        self.buffer = deque(maxlen=maxlen)
        self.size = maxlen

    def append(self, exp: Tuple[np.ndarray, Union[int, np.ndarray], float, np.ndarray, bool]):
        """Append an experience sample to the buffer.

        Args:
            exp: An experience sample consisting of state, action, reward, next_state and done.
        """
        self.buffer.append(exp)

    def __len__(self) -> int:
        """Get the buffer length.

        Returns:
            The buffer length.
        """
        return len(self.buffer)

    def sample(self, batch_size: int) -> List[np.ndarray]:
        """Sample a batch of replay experience from the buffer.

        Args:
            batch_size: Batch size.

        Returns:
            An array of state, action, reward, next_state and done arrays.

        Raises:
            AssertionError: Buffer contains less samples than the requested batch size.
        """
        assert (batch_size <= len(self.buffer))
        samples = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [np.array(x, dtype=np.float32) for x in zip(*[self.buffer[i] for i in samples])]

    def clear(self):
        """Clear the buffer from previous experience."""
        self.buffer.clear()
