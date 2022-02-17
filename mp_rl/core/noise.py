from abc import ABC, abstractmethod
from typing import Union

import numpy as np

T = Union[float, np.ndarray]


class NoiseProcess(ABC):
    """Abstract base class for NoiseProcesses.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def sample(self):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError


class OrnsteinUhlenbeckNoise(NoiseProcess):
    """Implements the Ornstein Uhlenbeck process to generate an N-dimensional noise.

    Noise sampled from this process is correlated with previous samples, which can be useful in RL
    applications.
    """

    def __init__(self, mu: T, sigma: T, dims: int):
        """Initializes the noise array.

        Args:
            mu (T): Previous noise reduction factor.
            sigma (T): Standard deviation of the gaussian noise in each step.
            dims (int): Noise dimension.
        """
        self.mu = mu
        self.sigma = sigma
        self.dims = dims
        self.noise = np.zeros(dims, dtype=np.float32)

    def sample(self) -> np.ndarray:
        """Samples from the noise process.

        Returns:
            A numpy array of noise that is correlated with previous samples.
        """
        self.noise = -self.noise * self.mu + self.sigma * np.random.randn(self.dims)
        return self.noise

    def reset(self):
        """Resets the noise process to start from 0.
        """
        self.noise[:] = 0


class GaussianNoise(NoiseProcess):
    """Standard gaussian noise.

    Samples are uncorrelated.
    """

    def __init__(self, mu: T, sigma: T, dims: int):
        """Initializes the parameters.

        Args:
            mu (T): Gaussian mean.
            sigma (T): Gaussian variance.
            dims (int): Noise dimension.
        """
        self.mu = mu
        self.sigma = sigma
        self.dims = dims

    def sample(self) -> np.ndarray:
        """Samples from the noise process.
        """
        return self.mu + self.sigma * np.random.randn(self.dims)

    def reset(self):
        """Gaussian noise is stateless, reset is a no-op."""
