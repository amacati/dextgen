import argparse
from pathlib import Path

import pytest
import gym
import yaml

import envs
from mp_rl.core.ddpg import DDPG


class TestDDPG:
    """Test DDPG learning on each environment for one epoch, one cycle.

    Since MPI is not easily supported within Docker, only the non-distributed case is tested.
    """

    @pytest.mark.parametrize("env", envs.available_envs)
    def test_ddpg(self, env):
        args = self.load_args(env)
        env = gym.make(args.env)
        ddpg = DDPG(env, args)
        ddpg.train()

    def load_args(self, env):
        args = argparse.Namespace()
        args.env = env
        path = Path(__file__).parents[1] / "config" / "experiment_config.yaml"
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

    @staticmethod
    def run_env(env_key):
        env = gym.make(env_key)
        env.reset()
        for _ in range(env._max_episode_steps):
            env.step(env.action_space.sample())
