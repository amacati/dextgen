"""Train an agent on an OpenAI gym environment with DDPG and PyTorch's DDP."""

import argparse
import logging
from pathlib import Path

import gym
import torch.multiprocessing as mp
import yaml

from mp_rl.core.ddpg import DDPG
from mp_rl.core.utils import init_process


def parse_args() -> argparse.Namespace:
    """Parse arguments for the gym environment and logging levels.

    Returns:
        The parsed arguments as a namespace.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",
                        help="Selects the gym environment",
                        choices=["FetchReach-v1", "FetchPickAndPlace-v1"],
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
        raise KeyError(f"Config file is missing config for env '{args.env}'")
    for key, val in config[args.env].items():
        setattr(args, key, val)


def launch_distributed_ddpg(args: argparse.Namespace):
    """Launch multiple training processes as a DataDistributedParallel group.

    Establishes the connection among processes for PyTorch and cleans up after processes exit.

    Args:
        args: User provided arguments namespace.
    """
    processes = []
    mp.set_start_method("spawn")
    # PyTorch averages gradients in DDP instead of accumulating them
    args.actor_lr *= args.nprocesses
    args.critic_rl *= args.nprocesses
    for rank in range(args.nprocesses):
        p = mp.Process(target=init_process,
                       args=(rank, args.nprocesses, loglvls[args.loglvl], run_dist_ddpg, args))
        p.start()
        processes.append(p)
    logging.info("Process spawn successful, awaiting join")
    for p in processes:
        p.join()
    logger.info("Processes joined, training complete.")


def run_dist_ddpg(rank: int, size: int, args: argparse.Namespace):
    """Start the training in distributed mode on a process of the DDP process group.

    Creates the gym and the DDPG module on each worker and starts the training.

    Args:
        rank: Process rank within the process group.
        size: Process group world size.
        args: User provided arguments namespace.
    """
    env = gym.make(args.env)
    ddpg = DDPG(env, args, world_size=size, rank=rank, dist=True)
    ddpg.train()


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
    if hasattr(args, "nprocesses") and args.nprocesses > 1:
        launch_distributed_ddpg(args)
    else:
        env = gym.make(args.env)
        ddpg = DDPG(env, args)
        ddpg.train()
