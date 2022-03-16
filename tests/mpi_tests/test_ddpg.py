import argparse
from pathlib import Path

import pytest
import gym
import yaml
from mpi4py import MPI

import envs
from mp_rl.core.ddpg import DDPG


@pytest.mark.mpi
def test_ddpg_mpi():
    comm = MPI.COMM_WORLD
    args = load_args("ShadowHandPickAndPlace-v0")
    env = gym.make(args.env)
    ddpg = DDPG(env, args, world_size=comm.Get_size(), rank=comm.Get_rank(), dist=True)
    ddpg.train()


def load_args(env):
    args = argparse.Namespace()
    args.env = env
    path = Path(__file__).parents[2] / "config" / "experiment_config.yaml"
    with open(path, "r") as f:
        config = yaml.load(f, yaml.SafeLoader)

    if "Default" not in config.keys():
        raise KeyError("Config file is missing required entry `Default`!")
    for key, val in config["Default"].items():
        setattr(args, key, val)

    if args.env in config.keys():
        for key, val in config[args.env].items():
            setattr(args, key, val)
    args.epochs = 1
    args.cycles = 1
    args.batches = 1
    args.evals = 1
    return args
