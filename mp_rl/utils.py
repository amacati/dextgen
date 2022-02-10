import os
import logging
from typing import Callable
import numpy as np
import torch.distributed as dist
import torch.nn as nn
import gym
from replay_buffer import MemoryBuffer


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
    target_state = target.state_dict()
    for k, v in network.state_dict().items():
        target_state[k] = (1 - tau)  * target_state[k]  + tau * v
    target.load_state_dict(target_state)
    return target


def running_average(values: list, window: int = 50, mode: str = 'valid') -> float:
    """Computes a running average over a list of values.
    
    Args:
        values (list): List of values that get smoothed.
        window (int, optional): Averaging window size.
        mode (str, optional): Modes for the convolution operation.
    """
    return np.convolve(values, np.ones(window)/window, mode=mode)


def fill_buffer(env: gym.Env, buffer: MemoryBuffer):
    """Fills the `buffer` with experiences under a uniformly random policy.
    
    Args:
        env (gym.Env): The gym environment.
        buffer (MemoryBuffer): Memory buffer which stores the experiences.
    """
    while len(buffer) < buffer.size:
        state = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            buffer.append((state, action, reward, next_state, done))
            state = next_state


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
    fn(rank, size, *args, **kwargs)