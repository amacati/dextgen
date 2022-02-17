from abc import ABC, abstractmethod
from typing import Callable, Union, Tuple
from collections import deque

import numpy as np


class ReplayBuffer(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def sample(self, *_):
        ...

    @abstractmethod
    def append(self, *_):
        ...

    @abstractmethod
    def __len__(self):
        ...


class TrajectoryBuffer:

    def __init__(self, size_s, size_a, size_g, T: int):
        self.T = T
        self.buffer = {
            "s": np.zeros([T + 1, size_s]),
            "a": np.zeros([T, size_a]),
            "g": np.zeros([T, size_g]),
            "ag": np.zeros([T + 1, size_g])
        }
        self.t = 0

    def append(self, *args):
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

    def __getitem__(self, key):
        return self.buffer[key]

    def __repr__(self):
        return repr(self.buffer)

    def __len__(self):
        return self.t

    def clear(self):
        self.buffer = {key: np.empty((self.T, size_b)) for key, size_b in self.buffer_sizes.items()}

    def keys(self):
        return self.buffer.keys()

    def values(self):
        return self.buffer.values()

    def items(self):
        return self.buffer.items()

    def __iter__(self):
        return self.buffer.__iter__(self)


class HERBuffer(ReplayBuffer):

    def __init__(self,
                 size_s: int,
                 size_a: int,
                 size_g: int,
                 T: int,
                 k: int,
                 max_samples: int,
                 reward_fun: Callable,
                 sample_mode: str = "her"):
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
        self._validate_episode(ep_buffer)
        idx = self.curr_size if self.curr_size < self.size else np.random.randint(0, self.size)
        for key, val in ep_buffer.items():  # Keys are already validated to match the buffer
            self.buffer[key][idx] = val
        self.curr_size = min(self.curr_size + 1, self.size)

    def _validate_episode(self, episode: TrajectoryBuffer):
        if len(episode) != self.T:
            raise RuntimeError("Episode length has to match time horizon T")

    def sample(self, N: int):
        if N > self.size * self.T:
            raise RuntimeError("Batch size N exceeds buffer contents")
        if self.sample_mode == "default":
            return default_sampling(self.buffer["s"][:self.curr_size],
                                    self.buffer["a"][:self.curr_size],
                                    self.buffer["g"][:self.curr_size],
                                    self.buffer["ag"][:self.curr_size], N, self.reward_fun)
        elif self.sample_mode == "her":
            return her_sampling(self.buffer["s"][:self.curr_size],
                                self.buffer["a"][:self.curr_size],
                                self.buffer["g"][:self.curr_size],
                                self.buffer["ag"][:self.curr_size], N, self.p_her, self.reward_fun)
        raise RuntimeError("Unsupported sample mode!")

    def _reward_fun(self, *_):
        raise NotImplementedError("Reward function has to be specified by user before use")

    def __len__(self):
        return self.curr_size

    def get_trajectory_buffer(self):
        return TrajectoryBuffer(self.size_s, self.size_a, self.size_g, self.T)


def her_sampling(states: np.ndarray, actions: np.ndarray, goals: np.ndarray, agoals: np.ndarray,
                 N: int, p_her: float, reward_fun: Callable) -> Tuple[np.ndarray]:
    assert states.ndim == 3, "Requires tensors of dimensions (episode, timestep, data_dim)"
    neps, T, _ = actions.shape
    e_idx = np.random.randint(0, neps, N)
    t_idx = np.random.randint(0, T, N)
    # Copy transitions to avoid changing the ground truth values after sampling
    goals = goals[e_idx, t_idx].copy()
    h_idx = np.where(np.random.uniform(size=N) < p_her)
    t_offset = (np.random.uniform(size=N) * (T - t_idx)).astype(int)
    f_idx = (t_idx + 1 + t_offset)[h_idx]
    goals[h_idx] = agoals[e_idx[h_idx], f_idx].copy()
    rewards = reward_fun(agoals[e_idx, t_idx + 1], goals, None)
    rewards = np.expand_dims(rewards, 1)
    next_states = states[e_idx, t_idx + 1].copy()
    states, actions = states[e_idx, t_idx].copy(), actions[e_idx, t_idx].copy()
    return states, actions, rewards, next_states, goals


def default_sampling(states: np.ndarray, actions: np.ndarray, goals: np.ndarray, agoals: np.ndarray,
                     N: int, reward_fun: Callable) -> Tuple[np.ndarray]:
    neps, T, _ = actions.shape
    e_idx = np.random.randint(0, neps, N)
    t_idx = np.random.randint(0, T, N)
    next_states, goals = states[e_idx, t_idx + 1].copy(), goals[e_idx, t_idx].copy()
    states, actions = states[e_idx, t_idx].copy(), actions[e_idx, t_idx].copy()
    rewards = np.expand_dims(reward_fun(agoals[e_idx, t_idx + 1], goals, None), 1)
    return states, actions, rewards, next_states, goals


class MemoryBuffer(ReplayBuffer):

    def __init__(self, maxlen: int = 10000):
        self.buffer = deque(maxlen=maxlen)
        self.size = maxlen

    def append(self, exp: Tuple[np.ndarray, Union[int, np.ndarray], float, np.ndarray, bool]):
        self.buffer.append(exp)

    def __len__(self):
        return len(self.buffer)

    def sample(self, n: int):
        assert (n <= len(self.buffer))
        samples = np.random.choice(len(self.buffer), n, replace=False)
        return [np.array(x, dtype=np.float32) for x in zip(*[self.buffer[i] for i in samples])]

    def clear(self):
        self.buffer.clear()
