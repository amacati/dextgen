import argparse
import logging
from pathlib import Path
import random

import gym
import numpy as np
import torch
import yaml

from mp_rl.core.ddpg import DDPG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="Selects the gym environment", choices=["FetchReach-v1"],
                        default="FetchReach-v1")
    parser.add_argument('--loglvl', help="Logger levels", choices=["DEBUG", "INFO", "WARN", "ERROR"],
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


def set_seeds(env, seed):
    env.seed(seed)
    obs = env.reset()  # Align random seeds with reference implementation
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    args = parse_args()
    expand_args(args)
    logger = logging.getLogger(__name__)
    loglvls = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARN": logging.WARN, "ERROR": logging.ERROR}
    logging.basicConfig()
    logging.getLogger().setLevel(loglvls[args.loglvl])
    env = gym.make(args.env)
    # set_seeds(env, 123)
    ddpg = DDPG(env, args)
    ddpg.train()