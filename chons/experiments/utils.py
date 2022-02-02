import numpy as np


def running_average(values: list, window: int = 50, mode: str = 'valid'):
    return np.convolve(values, np.ones(window)/window, mode=mode)
