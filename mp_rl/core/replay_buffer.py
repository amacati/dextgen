from abc import ABC, abstractmethod
from typing import Callable, Union, Tuple
from collections import deque, OrderedDict

import numpy as np
import torch


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
        self.buffer = {"s": np.zeros([T+1, size_s]),
                       "a": np.zeros([T, size_a]),
                       "g": np.zeros([T, size_g]),
                       "ag": np.zeros([T+1, size_g])}
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

    def __init__(self, size_s: int, size_a: int, size_g: int, T: int, k: int, max_samples: int,
                 reward_fun: Callable, sample_mode: str = "her"):
        self.size = max_samples // T
        self.size_s = size_s
        self.size_a = size_a
        self.size_g = size_g
        self.curr_size = 0
        self.T = T  # Episodes have a fixed time horizon T
        # Keys: state, action, goal, achieved goal
        self.buffer = {"s": np.zeros([self.size, T+1, size_s]),
                       "a": np.zeros([self.size, T, size_a]),
                       "g": np.zeros([self.size, T, size_g]),
                       "ag": np.zeros([self.size, T+1, size_g])}
        self.k = k
        self.p_her = 1 - 1./(1+k)
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
        if N > self.size*self.T:
            raise RuntimeError("Batch size N exceeds buffer contents")
        if self.sample_mode == "default":
            return self._sample_default(N)
        elif self.sample_mode == "her":
            return self._sample_her(N)
        raise RuntimeError("Unsupported sample mode!")

    def _sample_default(self, N: int):
        e_idx = np.random.randint(0, self.curr_size, N)
        t_idx = np.random.randint(0, self.T, N)
        # Only copies 's', 'a', 'g', 'ag'. Still missing 'agn', 'sn', 'r'
        transitions = {key: val[e_idx, t_idx].copy() for key, val in self.buffer.items()}
        transitions["sn"] = self.buffer["s"][e_idx, t_idx+1].copy()
        transitions["agn"] = self.buffer["ag"][e_idx, t_idx+1].copy()
        rewards = self.reward_fun(transitions["agn"], transitions["g"], None)
        transitions["r"] = np.expand_dims(rewards, 1)
        return (transitions[key] for key in ["s", "a", "r", "sn", "g"])

    def _sample_her(self, N: int):
        e_idx = np.random.randint(0, self.curr_size, N)
        t_idx = np.random.randint(0, self.T, N)
        # Copy transitions to avoid changing the ground truth values after sampling
        # Only copies 's', 'a', 'g', 'ag'. Still missing 'agn', 'sn', 'r'
        transitions = {key: val[e_idx, t_idx].copy() for key, val in self.buffer.items()}
        transitions["sn"] = self.buffer["s"][e_idx, t_idx+1].copy()
        transitions["agn"] = self.buffer["ag"][e_idx, t_idx+1].copy()
        h_idx = np.where(np.random.uniform(size=N) < self.p_her)
        t_offset = (np.random.uniform(size=N) * (self.T - t_idx)).astype(int)
        f_idx = (t_idx + 1 + t_offset)[h_idx]
        transitions["g"][h_idx] = self.buffer["ag"][e_idx[h_idx], f_idx].copy()
        rewards = self.reward_fun(transitions["agn"], transitions["g"], None)
        transitions["r"] = np.expand_dims(rewards, 1)
        return (transitions[key] for key in ["s", "a", "r", "sn", "g"])

    def _reward_fun(self, *_):
        raise NotImplementedError("Reward function has to be specified by user before use")

    def __len__(self):
        return self.curr_size

    def get_trajectory_buffer(self):
        return TrajectoryBuffer(self.size_s, self.size_a, self.size_g, self.T)


class MemoryBuffer(ReplayBuffer):

    def __init__(self, maxlen: int = 10000):
        self.buffer = deque(maxlen=maxlen)
        self.size = maxlen

    def append(self, exp: Tuple[np.ndarray, Union[int, np.ndarray], float, np.ndarray, bool]):
        self.buffer.append(exp)

    def __len__(self):
        return len(self.buffer)

    def sample(self, n: int):
        assert(n <= len(self.buffer))
        samples = np.random.choice(len(self.buffer), n, replace=False)
        return [np.array(x, dtype=np.float32) for x in zip(*[self.buffer[i] for i in samples])]

    def clear(self):
        self.buffer.clear()
