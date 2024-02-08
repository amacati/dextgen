"""Train an agent on an OpenAI gym environment with DDPG and PyTorch's DDP."""
import random
from pathlib import Path

import gymnasium
import fire
import yaml
import numpy as np
import torch
import wandb

from mp_rl.vec.ddpg import DDPG
from mp_rl.utils import WandBLogger, DummyWandBConfig


def set_seed(seed: int):
    """Set the random seed of all relevant modules for reproducible experiments.

    Args:
        env: Gym environment.
        seed: Seed used to set the seeds of all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def main(nruns: int = 1):
    env_name = "FetchPickAndPlace-v2"
    env_cfg = env_name.lower().replace("_", "-")[:-3] + "-vec-config.yaml"
    config_path = Path(__file__).parent / "config" / env_cfg
    cfg = load_config(config_path)  # Load a dummy WandB config for all ranks > 0

    save_path = Path(__file__).parent / "saves" / env_name
    save_path.mkdir(parents=True, exist_ok=True)

    num_envs = 32

    for i in range(nruns):
        # Sync vectorization mode for now as num_envs == 1
        env_kwargs = getattr(cfg, "env_kwargs", {})
        env_kwargs["num_envs"] = num_envs
        env = gymnasium.make_vec(env_name, **env_kwargs, vectorization_mode="async")
        eval_env = gymnasium.make_vec(env_name, **env_kwargs, vectorization_mode="async")
        if isinstance(cfg.seed, int):
            seed = cfg.seed + i * num_envs
            set_seed(seed)
        with wandb.init(project=env_name,
                        entity="amacati",
                        group=cfg.group if hasattr(cfg, "group") else None,
                        config=str(config_path),
                        save_code=True,
                        dir=save_path) as run:
            logger = WandBLogger(run)
            ddpg = DDPG(env, eval_env, run.config, logger, seed)
            ddpg.train()
            run.finish()


if __name__ == "__main__":
    fire.Fire(main)
