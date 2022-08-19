"""Noise process module.

To support stateful noise with possibly more complex sampling procedures, this module defines an
abstract :class:`.NoiseProcess` class that defines the noise sampling interface for the
:class:`mp_rl.core.actor.Actor`.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union

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

    def __init__(self, dims: int, mu: T, sigma: T, clip: Optional[float] = None):
        """Initialize the noise process.

        Args:
            dims: Noise dimension.
            mu: Previous noise reduction factor.
            sigma: Standard deviation of the gaussian noise in each step.
            clip: Optional noise clipping parameter.
        """
        self.mu = mu
        self.sigma = sigma
        self.dims = dims
        self.noise = np.zeros(dims, dtype=np.float32)
        self.clip = clip

    def sample(self) -> np.ndarray:
        """Sample from the noise process.

        Returns:
            A numpy array of noise that is correlated with previous samples.
        """
        self.noise -= self.noise * self.mu + np.random.normal(scale=self.sigma, size=self.dims)
        if self.clip is not None:
            self.noise = np.clip(self.noise, -self.clip, self.clip)
        return self.noise.copy()

    def reset(self):
        """Reset the internal noise process state to start from 0."""
        self.noise[:] = 0


class GaussianNoise(NoiseProcess):
    """Standard gaussian noise with uncorrelated samples."""

    def __init__(self, dims: int, mu: T, sigma: T):
        """Initialize the noise process.

        Args:
            dims: Noise dimension.
            mu: Gaussian mean.
            sigma: Gaussian variance.
        """
        self.dims = dims
        self.mu = mu
        self.sigma = sigma

    def sample(self) -> np.ndarray:
        """Sample from the noise process.

        Returns:
            A numpy array of gaussian noise.
        """
        return np.random.normal(self.mu, self.sigma, self.dims)

    def reset(self):
        """Gaussian noise is stateless, reset is a no-op."""


class UniformNoise(NoiseProcess):
    """Uniformly distributed noise with uncorrelated samples."""

    def __init__(self, dims: int):
        """Initialize the noise process.

        Args:
            dims: Noise dimension.
        """
        self.dims = dims

    def sample(self) -> np.ndarray:
        """Sample from the noise process.

        Returns:
            A numpy array of uniformly distributed noise.
        """
        return np.random.uniform(-1., 1., self.dims)

    def reset(self):
        """Uniform noise is stateless, reset is a no-op."""
