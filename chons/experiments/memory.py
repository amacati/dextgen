from collections import deque
from typing import Tuple, Union
import numpy as np

class MemoryBuffer:

    def __init__(self, maxlen: int = 10000):
        self.buffer = deque(maxlen=maxlen)

    def append(self, exp: Tuple[np.ndarray, Union[int, np.ndarray], float, np.ndarray, bool]):
        self.buffer.append(exp)
    
    def __len__(self):
        return len(self.buffer)

    def sample(self, n: int):
        assert(n <= len(self.buffer))
        samples = np.random.choice(len(self.buffer), n, replace=False)
        if 0 not in samples:
            samples[-1] = 0  # Always include the latest experience
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in samples])
        return (np.array(x, dtype=np.float32) for x in [states, actions, rewards, next_states, dones])
    
    def clear(self):
        self.buffer.clear()


class TrajectoryBuffer:

    def __init__(self):
        self.buffer = []

    def append(self, exp: Tuple[np.ndarray, Union[int, np.ndarray], float, np.ndarray, bool]):
        self.buffer.append(exp)
    
    def __len__(self):
        return len(self.buffer)

    def sample(self):
        states, actions, rewards, next_states, dones = zip(*self.buffer)
        return (np.array(x, dtype=np.float32) for x in [states, actions, rewards, next_states, dones])
    
    def clear(self):
        self.buffer.clear()