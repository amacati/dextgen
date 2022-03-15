"""Train an agent on an OpenAI gym environment with DDPG and PyTorch's DDP."""

import logging

import gym
from mpi4py import MPI

import envs  # Import registers environments with gym  # noqa: F401
from mp_rl.core.ddpg import DDPG
from parse_args import parse_args

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
    ddpg = DDPG(env, args, world_size=comm.Get_size(), rank=comm.Get_rank(), dist=True)
    ddpg.train()
