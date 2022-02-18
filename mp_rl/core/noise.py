"""Noise process classes that are used to sample possibly time correlated noise."""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np

T = Union[float, np.ndarray]


class NoiseProcess(ABC):
    """Abstract base class for NoiseProcesses."""

    def __init__(self):
        """Initialize the noise process."""
        super().__init__()

    @abstractmethod
    def sample(self) -> np.ndarray:
        """Sample a noise sample from the process.

        Returns:
            A noise sample array of the noise process' dimension.
        """

    @abstractmethod
    def reset(self):
        """Reset internal process states if any."""


class OrnsteinUhlenbeckNoise(NoiseProcess):
    """Ornstein Uhlenbeck noise process to generate an N-dimensional noise.

    Noise sampled from this process is correlated with previous samples, which can be useful in RL
    applications.
    """

    def __init__(self, mu: T, sigma: T, dims: int):
        """Initialize the noise process.

        Args:
            mu: Previous noise reduction factor.
            sigma: Standard deviation of the gaussian noise in each step.
            dims: Noise dimension.
        """
        self.mu = mu
        self.sigma = sigma
        self.dims = dims
        self.noise = np.zeros(dims, dtype=np.float32)

    def sample(self) -> np.ndarray:
        """Sample from the noise process.

        Returns:
            A numpy array of noise that is correlated with previous samples.
        """
        self.noise = -self.noise * self.mu + self.sigma * np.random.randn(self.dims)
        return self.noise

    def reset(self):
        """Reset the internal noise process state to start from 0."""
        self.noise[:] = 0


class GaussianNoise(NoiseProcess):
    """Standard gaussian noise with uncorrelated samples."""

    def __init__(self, mu: T, sigma: T, dims: int):
        """Initialize the noise process.

        Args:
            mu: Gaussian mean.
            sigma: Gaussian variance.
            dims: Noise dimension.
        """
        self.mu = mu
        self.sigma = sigma
        self.dims = dims

    def sample(self) -> np.ndarray:
        """Sample from the noise process.

        Returns:
            A numpy array of gaussian noise.
        """
        return self.mu + self.sigma * np.random.randn(self.dims)

    def reset(self):
        """Gaussian noise is stateless, reset is a no-op."""
