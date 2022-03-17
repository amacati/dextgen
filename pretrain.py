"""Pretrain grasping on a simple sphere before training in the actual environments."""
import logging
import argparse
from pathlib import Path
import inspect

import yaml
import gym
from mpi4py import MPI

from mp_rl.core.ddpg import DDPG
import envs
from envs.shadow_hand.shadowhand_pretrain import ShadowHandPretrain

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",
                        help="Selects the gym environment",
                        choices=[e for e in envs.available_envs if "ShadowHand" in e],
                        default="ShadowHandPickAndPlace-v0")
    parser.add_argument('--loglvl',
                        help="Logger levels",
                        choices=["DEBUG", "INFO", "WARN", "ERROR"],
                        default="INFO")
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    loglvls = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARN": logging.WARN,
        "ERROR": logging.ERROR
    }
    logging.basicConfig(level=loglvls[args.loglvl])

    path = Path(__file__).parent / "config" / "experiment_config.yaml"
    with open(path, "r") as f:
        config = yaml.load(f, yaml.SafeLoader)

    if "Default" not in config.keys():
        raise KeyError("Config file is missing required entry `Default`!")
    for key, val in config["Default"].items():
        setattr(args, key, val)

    if args.env not in config.keys():
        logger.info(f"No specific config for {args.env} found, using defaults for all settings.")
    else:
        for key, val in config[args.env].items():
            setattr(args, key, val)
    if "ShadowHandPretrain-v0" not in config.keys():
        logger.info("No specific config for ShadowHandPretain-v0 found.")
    else:
        for key, val in config["ShadowHandPretrain-v0"].items():
            if key == "kwargs":
                setattr(args, "kwargs", args.kwargs | val)  # Append kwargs
            else:
                setattr(args, key, val)  # Intentionally overwrites settings from args.env

    if hasattr(args, "kwargs") and args.kwargs:
        # Use kwargs for pretrain where applicable
        print(args.kwargs)
        pt_args = inspect.signature(ShadowHandPretrain)
        kwargs = {key: val for key, val in args.kwargs.items() if key in pt_args.parameters.keys()}
        print(kwargs)
        env = gym.make("ShadowHandPretrain-v0", **kwargs)
    else:
        env = gym.make("ShadowHandPretrain-v0")
    comm = MPI.COMM_WORLD
    ddpg = DDPG(env, args, world_size=comm.Get_size(), rank=comm.Get_rank(), dist=True)
    ddpg.train()

    path = Path(__file__).parent / "saves" / "pretrained" / args.env
    if ddpg.rank == 0:
        if not path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
        ddpg.save_models(path)
