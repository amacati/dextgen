from abc import ABC, abstractmethod
import numpy as np


class NoiseProcess(ABC):
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def sample(self):
        raise NotImplementedError
    
    @abstractmethod
    def reset(self):
        raise NotImplementedError


class OrnsteinUhlenbeckNoise(NoiseProcess):
    
    def __init__(self, mu, sigma, dims):
        self.mu = mu
        self.sigma = sigma
        self.dims = dims
        self.noise = np.zeros(dims, dtype=np.float32)
    
    def sample(self):
        self.noise = - self.noise * self.mu + self.sigma * np.random.randn(self.dims)
        return self.noise
    
    def reset(self):
        self.noise[:] = 0
        

class GaussianNoise(NoiseProcess):
    
    def __init__(self, mu, sigma, dims):
        self.mu = mu
        self.sigma = sigma
        self.dims = dims
        
    def sample(self):
        return self.mu + self.sigma * np.random.randn(self.dims)
    
    def reset(self):
        ...