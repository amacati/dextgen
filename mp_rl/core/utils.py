"""Utility functions for the mp_rl.core module."""

import os
import logging
from typing import Callable, Tuple, Any

import numpy as np
import torch
import torch.nn as nn

import torch.distributed as dist

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


def init_process(rank: int, size: int, loglvl: int, fn: Callable, *args: Any, **kwargs: Any):
    """Initialize a process with PyTorch's DDP process group and execute the provided function.

    Processes should target this function to ensure DDP is initialized before any calls that
    require the process group to be established.

    Args:
        rank: Process rank in the DDP process group.
        size: Total DDP world size.
        loglvl: Log level for Python's logging module in each process.
        fn: The main process function. Gets called after all initializations are done.
        args: Positional arguments for `fn`.
        kwargs: Keyword arguments for `fn`.
    """
    logging.basicConfig()
    logger.setLevel(loglvl)
    # Set environment variables required for DDP discovery service
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"  # 29500 in use on lsr.ei.tum clusters
    dist.init_process_group(backend="gloo", rank=rank, world_size=size)
    logger.info(f"P{rank}: Torch distributed process group established")
    torch.set_num_threads(2)  # Limit PyTorch's core count to avoid self contention
    fn(rank, size, *args, **kwargs)


def ddp_poll_shutdown(shutdown: bool = False) -> bool:
    """Poll a shutdown across the DDP process group.

    All processes send 0 for continue or 1 for shutdown. Each process performs an all_reduce op. If
    at least one process wants to shut down, the sum is > 0 and all processes receive the shutdown
    flag.

    Args:
        shutdown: True if shutdown is requested, else False.

    Returns:
        True if one process requested a shutdown, else False.
    """
    voting = torch.tensor([shutdown], dtype=torch.int8)
    dist.all_reduce(voting)  # All_reduce is in-place
    if voting.item() > 0:
        return True
    return False


def unwrap_obs(obs: dict) -> Tuple[np.ndarray]:
    """Unwrap an observation from a dictionary style OpenAI gym.

    Args:
        obs: Gym observation.

    Returns:
        A tuple of separated observation, desired goal and achieved goal arrays.
    """
    return obs["observation"], obs["desired_goal"], obs["achieved_goal"]
