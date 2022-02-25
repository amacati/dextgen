"""Normalizer class file."""

from typing import Union
from pathlib import Path

import pickle
import numpy as np
import torch
from mpi4py import MPI


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
        self.eps2 = np.ones(size, dtype=np.float32) * eps**2
        self.clip = clip
        # Tensors for allreduce ops to transfer stats between processe via MPI.
        # Local tensors hold stats from the current update, accumulate external stats in the
        # all_reduce phase, transfer the accumulated values into the all-time arrays and reset to
        # zero. Avoids including past stats from other processes as own values for the current run
        self.lsum = np.zeros(size, dtype=np.float32)
        self.lsum_sq = np.zeros(size, dtype=np.float32)
        self.lcount = np.zeros(1, dtype=np.float32)
        self.sum = np.zeros(size, dtype=np.float32)
        self.sum_sq = np.zeros(size, dtype=np.float32)
        self.count = np.zeros(1, dtype=np.float32)
        self.mean = np.zeros(size, dtype=np.float32)
        self.std = np.ones(size, dtype=np.float32)
        self.coal_buffer = np.zeros(size * 2 + 1, dtype=np.float32)
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
        return np.clip((x - self.mean) / self.std, -self.clip, self.clip)
        # TODO: return torch.clip((x - self.mean) / self.std, -self.clip, self.clip)

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
        self.lsum = np.sum(x, axis=0, dtype=np.float32)
        self.lsum_sq = np.sum(x**2, axis=0, dtype=np.float32)
        self.lcount[0] = x.shape[0]
        if self.dist:
            # Coalesce tensors to reduce communication overhead of all_reduce
            self.coal_buffer[0:self.size] = self.lsum
            self.coal_buffer[self.size:self.size * 2] = self.lsum_sq
            self.coal_buffer[self.size * 2] = self.lcount[0]
            MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, self.coal_buffer, op=MPI.SUM)
            self.lsum = self.coal_buffer[0:self.size]
            self.lsum_sq = self.coal_buffer[self.size:self.size * 2]
            self.lcount[0] = self.coal_buffer[self.size * 2]
        self._transfer_buffers()
        self.mean = self.sum / self.count
        self.std = (self.sum_sq / self.count - (self.sum / self.count)**2)
        np.maximum(self.eps2, self.std, out=self.std)  # Numeric stability
        np.sqrt(self.std, out=self.std)

    def init_dist(self):
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
