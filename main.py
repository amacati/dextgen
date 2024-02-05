"""Train an agent on an OpenAI gym environment with DDPG and PyTorch's DDP."""
import random
from pathlib import Path

import gymnasium
import fire
import yaml
import numpy as np
import torch
from mpi4py import MPI
import wandb

import envs  # Import registers environments with gym  # noqa: F401
from mp_rl.core.ddpg import DDPG
from mp_rl.utils import DummyLogger, WandBLogger, DummyWandBConfig


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


def main(env_name: str, nruns: int = 1):
    env_cfg = env_name.lower().replace("_", "-")[:-3] + "-config.yaml"
    config_path = Path(__file__).parent / "config" / env_cfg
    cfg = load_config(config_path)  # Load a dummy WandB config for all ranks > 0

    save_path = Path(__file__).parent / "saves" / env_name
    save_path.mkdir(parents=True, exist_ok=True)

    for i in range(nruns):
        env = gymnasium.make_vec(env_name, **getattr(cfg, "env_kwargs", {}))
        eval_env = gymnasium.make_vec(env_name, **getattr(cfg, "eval_env_kwargs", {}))
        comm = MPI.COMM_WORLD
        if cfg.seed:
            assert isinstance(cfg.seed, int)
            set_seed(env, cfg.seed + comm.Get_rank() + comm.Get_size() * i)
            set_seed(eval_env, cfg.seed + comm.Get_rank() + comm.Get_size() * i)
        if comm.Get_rank() == 0:
            with wandb.init(project=env_name,
                            entity="amacati",
                            group=cfg.group if hasattr(cfg, "group") else None,
                            config=str(config_path),
                            save_code=True,
                            dir=save_path) as run:
                logger = WandBLogger(run)
                ddpg = DDPG(env,
                            eval_env,
                            run.config,
                            logger,
                            world_size=comm.Get_size(),
                            rank=comm.Get_rank(),
                            dist=True)
                ddpg.train()
                run.finish()
        else:
            logger = DummyLogger()
            ddpg = DDPG(env,
                        eval_env,
                        cfg,
                        logger,
                        world_size=comm.Get_size(),
                        rank=comm.Get_rank(),
                        dist=True)
            ddpg.train()


if __name__ == "__main__":
    fire.Fire(main)
