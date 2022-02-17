import pickle
import numpy as np
import torch
import torch.distributed as dist


class Normalizer:

    def __init__(self, size, eps=1e-2, clip=np.inf):
        self.size = size
        self.eps2 = torch.ones(size, dtype=torch.float32,
                               requires_grad=False) * eps**2
        self.clip = clip
        # Tensors for allreduce ops to transfer stats between processe via torch dist and Gloo.
        # Local tensors hold stats from the current update, accumulate external stats in the
        # all_reduce phase, transfer the accumulated values into the all-time tensors and reset to
        # zero. Avoids including past stats from other processes as own values for the current run
        self.lsum = torch.zeros(size, dtype=torch.float32, requires_grad=False)
        self.lsum_sq = torch.zeros(size,
                                   dtype=torch.float32,
                                   requires_grad=False)
        self.lcount = torch.zeros(1, dtype=torch.float32, requires_grad=False)
        self.sum = torch.zeros(size, dtype=torch.float32, requires_grad=False)
        self.sum_sq = torch.zeros(size,
                                  dtype=torch.float32,
                                  requires_grad=False)
        self.count = torch.zeros(1, dtype=torch.float32, requires_grad=False)
        self.mean = torch.zeros(size, dtype=torch.float32, requires_grad=False)
        self.std = torch.ones(size, dtype=torch.float32, requires_grad=False)
        self.dist = False

    def __call__(self, x: np.ndarray):
        return self.normalize(x)

    def normalize(self, x) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            return np.clip((x - self.mean.numpy()) / self.std.numpy(),
                           -self.clip, self.clip)
        return torch.clip((x - self.mean) / self.std, -self.clip, self.clip)

    def update(self, x: np.ndarray):
        assert x.ndim == 2, "Expecting batches of states as updates, not single states!"
        x = torch.as_tensor(x)
        self.lsum = torch.sum(x, dim=0, dtype=torch.float32)
        self.lsum_sq = torch.sum(x.pow(2), dim=0, dtype=torch.float32)
        self.lcount[0] = x.shape[0]
        if self.dist:
            # Coalesce tensors to reduce communication overhead of all_reduce. In-place op
            dist.all_reduce_coalesced([self.lsum, self.lsum_sq, self.lcount])
        self._transfer_buffers()
        self.mean = self.sum / self.count
        self.std = (self.sum_sq / self.count - (self.sum / self.count).pow(2))
        torch.maximum(self.eps2, self.std, out=self.std)  # Numeric stability
        torch.sqrt(self.std, out=self.std)

    def init_ddp(self):
        self.dist = True

    def _transfer_buffers(self):
        self.sum += self.lsum
        self.sum_sq += self.lsum_sq
        self.count += self.lcount
        self.lsum[:] = 0  # Reset local tensors to not sum up previous runs
        self.lsum_sq[:] = 0
        self.lcount[:] = 0

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
