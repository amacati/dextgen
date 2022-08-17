"""Normalizer module.

Enables distributed normalizers to keep preprocessing consistent across all nodes. The normalizer is
based on the implementation in https://github.com/openai/baselines.
"""
from __future__ import annotations
from typing import Optional

import pickle
import numpy as np
from mpi4py import MPI

from mp_rl.utils import import_guard

if import_guard():
    from pathlib import Path  # noqa: TC003, is guarded


class Normalizer:
    """Normalizer to maintain an estimate on the current data mean and variance.

    Used to normalize input data to zero mean and unit variance. Supports synchronization over
    multiple processes via MPI.
    """

    def __init__(self,
                 size: int,
                 world_size: int,
                 eps: float = 1e-2,
                 clip: float = np.inf,
                 idx: Optional[np.ndarray] = None):
        """Initialize local and global buffer arrays for distributed mode.

        Args:
            size: Data dimension. Each dimensions mean and variance is tracked individually.
            world_size: MPI communication group size.
            eps: Minimum variance value to ensure numeric stability. Has to be larger than 0.
            clip: Clipping value for normalized data.
            idx: Optional index list that selects which entries should be normalized.
        """
        self.size = size
        self.world_size = world_size
        self.eps2 = np.ones(size, dtype=np.float32) * eps**2
        self.clip = clip
        # Arrays for allreduce ops to transfer stats between processe via MPI. Local arrays hold
        # stats from the current update, accumulate external stats in the all_reduce phase, transfer
        # the accumulated values into the all-time arrays and reset to zero. This avoids including
        # past stats from other processes as own values for the current run
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
        self.idx = idx

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Alias for `self.normalize`."""
        return self.normalize(x)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize the input data with the current mean and variance estimate.

        If the normalizer was initialized with an index array, only the entries corresponding to the
        indices are normalized. The rest of the array is returned unchanged.

        Args:
            x: Input data array.

        Returns:
            The normalized data.
        """
        if self.idx is not None:
            _x = x.copy()  # Make sure the input array remains unchanged
            _norm = (x[..., self.idx] - self.mean[..., self.idx]) / self.std[..., self.idx]
            _x[..., self.idx] = np.clip(_norm, -self.clip, self.clip)
            return _x
        return np.clip((x - self.mean) / self.std, -self.clip, self.clip)

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

    def load(self, path: Path):
        """Load parameters from a pickle save of a normalizer.

        Args:
            path: Savefile path.
        """
        with open(path, "rb") as f:
            normalizer = pickle.load(f)
        assert normalizer.size == self.size
        for attr in ["eps2", "lsum", "lsum_sq", "lcount", "sum", "sum_sq", "count", "mean", "std"]:
            setattr(self, attr, getattr(normalizer, attr))
