"""Utility module.

MPI network parameter updates are based on OpenAI's baselines.
See https://github.com/openai/baselines.
"""
import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from mpi4py import MPI

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


###############################
# Taken from OpenAI baselines #
###############################
def sync_networks(network: nn.Module):
    """Synchronize networks across MPI workers by broadcasting the weights of process 0.

    Note:
        In-place function.

    Args:
        network: PyTorch module that is synchronized across all workers.
    """
    flat_params = _get_flat_params_or_grads(network, mode='params')
    MPI.COMM_WORLD.Bcast(flat_params, root=0)
    _set_flat_params_or_grads(network, flat_params, mode='params')


def sync_grads(network: nn.Module):
    """Synchronize gradiends across MPI workers by allreduce.

    Note:
        In-place function.

    Args:
        network: PyTorch module that synchronizes the parameter gradients.
    """
    flat_grads = _get_flat_params_or_grads(network, mode='grads')
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, flat_grads, op=MPI.SUM)
    _set_flat_params_or_grads(network, flat_grads, mode='grads')


def _get_flat_params_or_grads(network: nn.Module, mode: str = "params") -> np.ndarray:
    """Create a coalesced numpy array from the network parameters.

    Args:
        network: PyTorch module from which parameters are gathered.
        mode: Toggles between weights and gradients. Either `data` or `grad`.

    Returns:
        A coalesced array of the network parameters.
    """
    attr = 'data' if mode == 'params' else 'grad'
    return np.concatenate(
        [getattr(param, attr).numpy().flatten() for param in network.parameters()])


def _set_flat_params_or_grads(network: nn.Module, flat_params: np.ndarray, mode: str = "params"):
    """Set the network parameters from a coalesced numpy array.

    Args:
        network: PyTorch module that gets overwritten.
        flat_params: Flattened parameters as numpy array.
        mode: Toggles between weights and gradients. Either `data` or `grad`.
    """
    attr = 'data' if mode == 'params' else 'grad'
    pointer = 0
    for param in network.parameters():
        getattr(param, attr).copy_(
            torch.tensor(flat_params[pointer:pointer + param.data.numel()]).view_as(param.data))
        pointer += param.data.numel()
