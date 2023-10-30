"""Utility module.

MPI network parameter updates are based on OpenAI's baselines.
See https://github.com/openai/baselines.
"""
import logging
from typing import Tuple

import numpy as np
import torch.nn as nn

logger = logging.getLogger(__name__)


def soft_update(network: nn.Module, target: nn.Module, tau: float) -> nn.Module:
    """Perform a soft update of the target network's weights.

    Shifts the weights of the ``target`` by a factor of ``tau`` into the direction of the
    ``network``.

    Args:
        network: Network from which to copy the weights.
        target: Network that gets updated.
        tau: Controls how much the weights are shifted. Valid in [0, 1].

    Returns:
        The updated target network.
    """
    for network_p, target_p in zip(network.parameters(), target.parameters()):
        target_p.data.copy_(tau * network_p.data + (1 - tau) * target_p)
    return target


def running_average(values: list, window: int = 50, mode: str = 'valid') -> float:
    """Compute a running average over a list of values.

    Args:
        values: List of values that get smoothed.
        window: Averaging window size.
        mode: Modes for the convolution operation.
    """
    return np.convolve(values, np.ones(window) / window, mode=mode)


def unwrap_obs(obs: dict) -> Tuple[np.ndarray]:
    """Unwrap an observation from a dictionary style OpenAI gym.

    Args:
        obs: Gym observation.

    Returns:
        A tuple of separated observation, desired goal and achieved goal arrays.
    """
    return obs["observation"], obs["desired_goal"], obs["achieved_goal"]
