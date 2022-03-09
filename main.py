"""Train an agent on an OpenAI gym environment with DDPG and PyTorch's DDP."""

import argparse
import logging
from pathlib import Path

import gym
import yaml
from mpi4py import MPI

import envs  # Import registers environments with gym  # noqa: F401
from mp_rl.core.ddpg import DDPG


def parse_args() -> argparse.Namespace:
    """Parse arguments for the gym environment and logging levels.

    Returns:
        The parsed arguments as a namespace.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",
                        help="Selects the gym environment",
                        choices=[
                            "FetchReach-v1", "FetchPickAndPlace-v1", "ObstacleReach-v0",
                            "UnevenPickAndPlace-v0", "SeaClearPickAndPlace-v0",
                            "SizePickAndPlace-v0", "ShadowHandPickAndPlace-v0",
                            "OrientPickAndPlace-v0", "ShadowHandEigengrasps-v0"
                        ],
                        default="FetchReach-v1")
    parser.add_argument('--loglvl',
                        help="Logger levels",
                        choices=["DEBUG", "INFO", "WARN", "ERROR"],
                        default="INFO")
    args = parser.parse_args()
    return args


def expand_args(args: argparse.Namespace):
    """Expand the arguments namespace with settings from the main config file.

    Config can be found at './config/experiment_config.yaml'. Each config must be named after their
    gym name.

    Args:
        args: User provided arguments namespace.
    """
    path = Path(__file__).parent / "config" / "experiment_config.yaml"
    with open(path, "r") as f:
        config = yaml.load(f, yaml.SafeLoader)

    if args.env not in config.keys():
        raise KeyError(f"Environment config file is missing config for env '{args.env}'")
    for key, val in config[args.env].items():
        setattr(args, key, val)


if __name__ == "__main__":
    args = parse_args()
    expand_args(args)
    logger = logging.getLogger(__name__)
    loglvls = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARN": logging.WARN,
        "ERROR": logging.ERROR
    }
    logging.basicConfig()
    logging.getLogger().setLevel(loglvls[args.loglvl])
    if hasattr(args, "kwargs") and args.kwargs:
        env = gym.make(args.env, **args.kwargs)
    else:
        env = gym.make(args.env)
    comm = MPI.COMM_WORLD
    ddpg = DDPG(env, args, world_size=comm.Get_size(), rank=comm.Get_rank(), dist=True)
    ddpg.train()
