import argparse
import logging
from pathlib import Path

import gym
import torch.multiprocessing as mp
import yaml

from mp_rl.core.ddpg import DDPG
from mp_rl.core.utils import init_process


def parse_args() -> argparse.Namespace:
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


def expand_args(args):
    path = Path(__file__).parent / "config" / "experiment_config.yaml"
    with open(path, "r") as f:
        config = yaml.load(f, yaml.SafeLoader)

    if args.env not in config.keys():
        raise KeyError(f"Config file is missing config for env '{args.env}'")
    for key, val in config[args.env].items():
        setattr(args, key, val)


def launch_distributed_ddpg(args):
    processes = []
    mp.set_start_method("spawn")
    for rank in range(args.nprocesses):
        p = mp.Process(target=init_process,
                       args=(rank, args.nprocesses, loglvls[args.loglvl],
                             run_dist_ddpg, args))
        p.start()
        processes.append(p)
    logging.info("Process spawn successful, awaiting join")
    for p in processes:
        p.join()
    logger.info("Processes joined, training complete.")


def run_dist_ddpg(rank, size, args):
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
