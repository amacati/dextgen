"""Normalizer class file."""

from typing import Union
from pathlib import Path

import pickle
import numpy as np
import torch
import torch.distributed as dist


class Normalizer:
    """Normalizer to maintain an estimate on the current data mean and variance.

    Used to normalize input data to zero mean and unit variance. Supports synchronization over
    multiple processes via torch distributed.
    """

    def __init__(self, size: int, world_size: int, eps: float = 1e-2, clip: float = np.inf):
        """Initialize local and global buffer tensors for distributed mode.

        Args:
            size: Data dimension. Each dimensions mean and variance is tracked individually.
            world_size: Torch distributed communication group size.
            eps: Minimum variance value to ensure numeric stability. Has to be larger than 0.
            clip: Clipping value for normalized data.
        """
        self.size = size
        self.world_size = world_size
        self.eps2 = torch.ones(size, dtype=torch.float32, requires_grad=False) * eps**2
        self.clip = clip
        # Tensors for allreduce ops to transfer stats between processe via torch dist and Gloo.
        # Local tensors hold stats from the current update, accumulate external stats in the
        # all_reduce phase, transfer the accumulated values into the all-time tensors and reset to
        # zero. Avoids including past stats from other processes as own values for the current run
        self.lsum = torch.zeros(size, dtype=torch.float32, requires_grad=False)
        self.lsum_sq = torch.zeros(size, dtype=torch.float32, requires_grad=False)
        self.lcount = torch.zeros(1, dtype=torch.float32, requires_grad=False)
        self.sum = torch.zeros(size, dtype=torch.float32, requires_grad=False)
        self.sum_sq = torch.zeros(size, dtype=torch.float32, requires_grad=False)
        self.count = torch.ones(1, dtype=torch.float32, requires_grad=False)
        self.mean = torch.zeros(size, dtype=torch.float32, requires_grad=False)
        self.std = torch.ones(size, dtype=torch.float32, requires_grad=False)
        self.dist = False

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Alias for `self.normalize`."""
        return self.normalize(x)

    def normalize(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Normalize the input data with the current mean and variance estimate.

        Args:
            x: Input data array. Supports both numpy arrays and torch Tensors.

        Returns:
            The normalized data. Preserves input data type.
        """
        if isinstance(x, np.ndarray):
            return np.clip((x - self.mean.numpy()) / self.std.numpy(), -self.clip, self.clip)
        return torch.clip((x - self.mean) / self.std, -self.clip, self.clip)

    def update(self, x: np.ndarray):
        """Update the mean and variance estimate with new data.

        If distributed mode is activated, local buffers perform an allreduce op before being added
        to the global estimate. For the reasoning behind separate buffers see `__init__` comments.

        Args:
            x: New input data. Expects a 3D array of shape (episodes, timestep, data dimension).

        Raises:
            AssertionError: Shape check failed.
        """
        assert x.ndim != 3, "Expecting 3D arrays of shape (episodes, timestep, data dimension)!"
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
        """Initialize distributed mode."""
        self.dist = True

    def _transfer_buffers(self):
        """Add the local buffers to the global estimate and reset the buffers.

        Average before summing to normalizer.
        """
        self.sum += self.lsum / self.world_size
        self.sum_sq += self.lsum_sq / self.world_size
        self.count += self.lcount / self.world_size
        self.lsum[:] = 0  # Reset local tensors to not sum up previous runs
        self.lsum_sq[:] = 0
        self.lcount[:] = 0

    def save(self, path: Path):
        """Save the normalizer as a pickle object.

        Args:
            path: Savefile path.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)
