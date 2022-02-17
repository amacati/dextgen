import os
import logging
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import matplotlib.pyplot as plt
import json

logger = logging.getLogger(__name__)


def soft_update(network: nn.Module, target: nn.Module, tau: float) -> nn.Module:
    """Performs a soft update of the target network's weights.

    Shifts the weights of the ``target`` by a factor of ``tau`` into the direction of the
    ``network``.

    Args:
        network (nn.Module): Network from which to copy the weights.
        target (nn.Module): Network that gets updated.
        tau (float): Controls how much the weights are shifted. Valid in [0, 1].

    Returns:
        target (nn.Module): The updated target network.
    """
    for network_p, target_p in zip(network.parameters(), target.parameters()):
        target_p.data.copy_(tau * network_p.data + (1 - tau) * target_p)
    return target


def running_average(values: list, window: int = 50, mode: str = 'valid') -> float:
    """Computes a running average over a list of values.

    Args:
        values (list): List of values that get smoothed.
        window (int, optional): Averaging window size.
        mode (str, optional): Modes for the convolution operation.
    """
    return np.convolve(values, np.ones(window) / window, mode=mode)


def init_process(rank: int, size: int, loglvl: int, fn: Callable, *args, **kwargs):
    """Initializes a process with PyTorch's DDP process group and executes the provided function.

    Processes should target this function to ensure DDP is initialized before any calls that
    require the process group to be established.

    Args:
        rank (int): Process rank in the DDP process group.
        size (int): Total DDP world size.
        loglvl (int): Log level for Python's logging module in each process.
        fn (Callable): The main process function. Gets called after all initializations are done.
        args (any): Positional arguments for `fn`.
        kwargs (any): Keyword arguments for `fn`.
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


def ddp_poll_shutdown(shutdown: bool = False):
    """Synchronized shutdown poll across DDP process group.

    All processes send 0 for continue or 1 for shutdown. Each process performs an all_reduce op. If
    at least one process wants to shut down, the sum is > 0 and all processes abort.

    Args:
        shutdown (bool): True if shutdown is requested, else False.

    Returns:
        True if one process requested a shutdown, else False.
    """
    voting = torch.tensor([shutdown], dtype=torch.int8)
    dist.all_reduce(voting)  # All_reduce is in-place
    if voting.item() > 0:
        return True
    return False


def save_plots(rewards: list[float], ep_len: list[float], path: Path, window: int = 10):
    fig, ax = plt.subplots(1, 2, figsize=(15, 4))
    ax[0].plot(rewards)
    smooth_reward = running_average(rewards, window=window)
    index = range(len(rewards) - len(smooth_reward), len(rewards))
    ax[0].plot(index, smooth_reward)
    ax[0].set_xlabel('Episode')
    ax[0].set_ylabel('Accumulated reward')
    ax[0].set_title('Agent reward over time')
    ax[0].legend(["Episode reward", "Running average reward"])

    ax[1].plot(ep_len)
    smooth_len = running_average(ep_len, window=window)
    index = range(len(ep_len) - len(smooth_len), len(rewards))
    ax[1].plot(index, smooth_len)
    ax[1].set_xlabel('Episode')
    ax[1].set_ylabel('Episode length')
    ax[1].set_title('Episode timestep development')
    ax[1].legend(["Episode length", "Running average length"])
    plt.savefig(path)


def save_stats(rewards: list[float], ep_len: list[float], path: Path, window: int = 10):
    smooth_reward = running_average(rewards, window=window)
    smooth_len = running_average(ep_len, window=window)
    stats = {
        "final_reward": rewards[-1],
        "final_av_reward": smooth_reward[-1],
        "final_ep_len": ep_len[-1],
        "final_ep_av_len": smooth_len[-1]
    }
    with open(path, "w") as f:
        json.dump(stats, f)


def unwrap_obs(obs: dict) -> Tuple[np.ndarray]:
    return obs["observation"], obs["desired_goal"], obs["achieved_goal"]
