"""Train an agent on an OpenAI gym environment with DDPG and PyTorch's DDP."""
import random
from pathlib import Path

import gymnasium
import fire
import yaml
import numpy as np
import torch
import wandb

import envs  # Import registers environments with gym  # noqa: F401
from mp_rl.core.ddpg import DDPG
from mp_rl.utils import WandBLogger, DummyWandBConfig


def set_seed(env: gymnasium.Env, seed: int):
    """Set the random seed of all relevant modules for reproducible experiments.

    Args:
        env: Gym environment.
        seed: Seed used to set the seeds of all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)


def load_config(path: Path):
    """Load the config file from the given path.

    Args:
        path: Path to the config file.

    Returns:
        The config as a dummy WandB config.
    """
    with open(path, "r") as f:
        config = yaml.load(f, yaml.SafeLoader)
    return DummyWandBConfig(config)


def main(env_name: str):
    env_cfg = env_name.lower().replace("_", "-")[:-3] + "-config.yaml"
    config_path = Path(__file__).parent / "config" / env_cfg
    save_path = Path(__file__).parent / "saves" / env_name

    with wandb.init(project=env_name,
                    entity="amacati",
                    config=str(config_path),
                    save_code=True,
                    dir=save_path) as run:

        save_path = Path(__file__).parent / "saves" / env_name
        save_path.mkdir(parents=True, exist_ok=True)
        kwargs = getattr(run.config, "env_kwargs", {})
        env = gymnasium.make(env_name, **kwargs)
        assert isinstance(run.config.seed, int)
        set_seed(env, run.config.seed)
        logger = WandBLogger(run)
        ddpg = DDPG2(env, run.config, logger)
        ddpg.train()


if __name__ == "__main__":
    fire.Fire(main)
