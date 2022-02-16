import time

import pickle
import numpy as np
import torch
import torch.distributed as dist


class Normalizer:
    
    def __init__(self, size, eps=1e-2, clip=np.inf):
        self.size = size
        self.eps2 = torch.ones(size, dtype=torch.float32, requires_grad=False)*eps**2
        self.clip = clip
        self.cont_stats = torch.zeros((size, 3), dtype=torch.float32, requires_grad=False)
        self.sum = torch.zeros(size, dtype=torch.float32, requires_grad=False)
        self.sum_sq = torch.zeros(size, dtype=torch.float32, requires_grad=False)
        self.count = torch.ones(1, dtype=torch.float32, requires_grad=False)
        self.mean = torch.zeros(size, dtype=torch.float32, requires_grad=False)
        self.std = torch.ones(size, dtype=torch.float32, requires_grad=False)
        self.dist = False

    def __call__(self, x: np.ndarray):
        return self.normalize(x)

    def normalize(self, x) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            return np.clip((x - self.mean.numpy()) / self.std.numpy(), -self.clip, self.clip)
        return torch.clip((x - self.mean) / self.std, -self.clip, self.clip)

    def update(self, x: np.ndarray):
        assert x.ndim == 2, "Expecting batches of states as updates, not single states!"
        x = torch.as_tensor(x)
        self.sum += torch.sum(x, dim=0)
        self.sum_sq += torch.sum(x.pow(2), dim=0)
        self.count += x.shape[0]
        if self.dist:
            self._sync_params()
        self.mean = self.sum / self.count
        self.std = (self.sum_sq / self.count - (self.sum/self.count).pow(2))
        torch.maximum(self.eps2, self.std, out=self.std)  # Numeric stability
        torch.sqrt(self.std, out=self.std)

    def _sync_params(self):
        t1 = time.perf_counter()
        dist.all_reduce(self.sum)
        dist.all_reduce(self.sum_sq)
        dist.all_reduce(self.count)
        t2 = time.perf_counter()
        print(f"_sync_params: {t2-t1}s")
    
    def _sync_params_experimental(self):
        t1 = time.perf_counter()
        dist.all_reduce(self.cont_stats)
        t2 = time.perf_counter()
        print(f"_sync_params_experimental: {t2-t1}s")

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)