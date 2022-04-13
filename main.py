"""Train an agent on an OpenAI gym environment with DDPG and PyTorch's DDP."""

import logging
import random
from pathlib import Path

import gym
import numpy as np
import torch
from mpi4py import MPI

import envs  # Import registers environments with gym  # noqa: F401
from mp_rl.core.ddpg import DDPG
from parse_args import parse_args


def set_seed(env: gym.Env, seed: int):
    """Set the random seed of all relevant modules for reproducible experiments.

    Args:
        env: Gym environment.
        seed: Seed used to set the seeds of all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)


if __name__ == "__main__":
    args = parse_args()
    logger = logging.getLogger(__name__)
    loglvls = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARN": logging.WARN,
        "ERROR": logging.ERROR
    }
    logging.basicConfig(level=loglvls[args.loglvl])
    if hasattr(args, "kwargs") and args.kwargs:
        env = gym.make(args.env, **args.kwargs)
    else:
        env = gym.make(args.env)
    comm = MPI.COMM_WORLD
    if args.seed:
        assert isinstance(args.seed, int)
        set_seed(env, args.seed + comm.Get_rank())
    ddpg = DDPG(env, args, world_size=comm.Get_size(), rank=comm.Get_rank(), dist=True)
    if args.load_pretrained:
        path = Path(__file__).parent / "saves" / "pretrain" / env.gripper_type
        logger.info(f"Loading pretrained DDPG model from {path}")
        ddpg.load_pretrained(path)
    ddpg.train()
