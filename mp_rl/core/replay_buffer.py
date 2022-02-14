from abc import ABC, abstractmethod
from typing import Callable, Optional, Union, Tuple
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


class HERBuffer(ReplayBuffer):

    def __init__(self, size_s: int, size_a: int, size_g: int, T: int, k: int, max_samples: int,
                 sample_mode: str = "her", reward_fun: Optional[Callable] = None):
        self.size = max_samples // T
        self.curr_size = 0
        self.T = T  # Episodes have a fixed time horizon T
        # Keys: state, action, next state, reward, done, goal desired, goal achieved
        self.buffer_sizes = {"s": size_s, "a": size_a, "sn": size_s, "r": 1, "d": 1, "g": size_g,
                             "ag": size_g}
        self.buffer_keys = ["s", "a", "sn", "r", "d", "g", "ag"]  # Order required for sampling
        self.buffer = {key: np.zeros([self.size, T, size_b])
                       for key, size_b in self.buffer_sizes.items()}
        self.k = k
        self.p_her = 1 - 1./(1+k)
        assert sample_mode in ["her", "default"]
        self.sample_mode = sample_mode
        self.reward_fun = reward_fun or self._reward_fun

    def append(self, episode: dict):
        self._validate_episode(episode)
        idx = self.curr_size if self.curr_size < self.size else np.random.randint(0, self.size)
        for key, val in episode.items():  # Keys are already validated to match the buffer
            self.buffer[key][idx] = val
        self.curr_size = min(self.curr_size + 1, self.size)

    def _validate_episode(self, episode: dict):
        if not set(self.buffer_keys) == episode.keys():
            raise KeyError("Episode keys do not match buffer keys or contains unknown keys!")
        if len(episode["s"]) != self.T:
            raise IndexError("Episode length has to match time horizon T")

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
        return {key: val[e_idx, t_idx].copy() for key, val in self.buffer.items()}

    def _sample_her(self, N: int):
        e_idx = np.random.randint(0, self.curr_size, N)
        t_idx = np.random.randint(0, self.T, N)
        # Copy transitions to avoid changing the ground truth values after sampling
        transitions = {key: val[e_idx, t_idx].copy() for key, val in self.buffer.items()}
        h_idx = np.where(np.random.uniform(size=N) < self.p_her)
        t_offset = (np.random.uniform(size=N) * (self.T - t_idx)).astype(int)
        f_idx = np.minimum(t_idx + 1 + t_offset, self.T-1)[h_idx]  # Force future goals w/o overflow
        transitions["g"][h_idx] = self.buffer["ag"][e_idx[h_idx], f_idx]
        transitions["r"] = self.reward_fun(transitions)
        return transitions

    def _reward_fun(self, *_):
        raise NotImplementedError("Reward function has to be specified by user before use")

    def __len__(self):
        return self.curr_size

    def get_trajectory_buffer(self):
        return {key: np.zeros((self.T, size_b)) for key, size_b in self.buffer_sizes.items()}


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
