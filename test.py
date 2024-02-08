"""Test a previously trained agent on an OpenAI gym environment."""
import logging
from pathlib import Path
import random
import time

import pickle
import torch
import gymnasium
import fire
import yaml
import numpy as np
from tqdm import tqdm

from mp_rl.core.utils import unwrap_obs
from mp_rl.core.actor import DDP
from mp_rl.utils import DummyWandBConfig


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


def set_seed(seed: int):
    """Set the random seed of all relevant modules for reproducible experiments.

    Args:
        env: Gym environment.
        seed: Seed used to set the seeds of all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def main(env_name: str = "FetchPickAndPlace-v2", nruns: int = 1):
    env_cfg = env_name.lower().replace("_", "-")[:-3] + "-config.yaml"
    config_path = Path(__file__).parent / "config" / env_cfg
    cfg = load_config(config_path)  # Load a dummy WandB config for all ranks > 0

    save_path = Path(__file__).parent / "saves" / env_name
    save_path.mkdir(parents=True, exist_ok=True)

    tstart = time.perf_counter()
    num_envs = 2
    env_kwargs = getattr(cfg, "env_kwargs", {})
    env_kwargs["num_envs"] = num_envs
    env = gymnasium.make_vec(env_name, num_envs=num_envs, vectorization_mode="sync")

    # Load the actor network and normalization parameters
    size_g = len(env.single_observation_space["desired_goal"].low)
    size_s = len(env.single_observation_space["observation"].low) + size_g
    size_a = len(env.single_action_space.low)

    actor = DDP(size_s, size_a, cfg.actor_net_nlayers, cfg.actor_net_layer_width)
    path = Path(__file__).parent / "saves" / env_name
    actor.load_state_dict(torch.load(path / "actor.pt"))
    with open(path / "state_norm.pkl", "rb") as f:
        state_norm = pickle.load(f)
    with open(path / "goal_norm.pkl", "rb") as f:
        goal_norm = pickle.load(f)

    success = 0.
    for i in tqdm(range(nruns)):
        seed = cfg.seed + i * num_envs
        set_seed(seed)
        state, goal, _ = unwrap_obs(env.reset(seed=seed)[0])
        for _ in range(env.spec.max_episode_steps):
            state_n, goal = state_norm(state), goal_norm(goal)
            state_n = torch.as_tensor(state_n, dtype=torch.float32)
            goal = torch.as_tensor(goal, dtype=torch.float32)
            # action = env.action_space.sample()
            action = actor(torch.cat([state_n, goal], dim=-1)).numpy()
            next_obs, reward, _, _, info = env.step(action)
            state, goal, _ = unwrap_obs(next_obs)

        success += np.sum(reward == 0.)
    env.close()
    t_elapsed = time.perf_counter() - tstart
    print(f"Agent success rate: {success/(nruns * num_envs):.2f}")
    print(f"Elapsed time: {t_elapsed:.2f} s")


if __name__ == "__main__":
    logger = logging.getLogger("GymTestScript")
    loglvls = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARN": logging.WARN,
        "ERROR": logging.ERROR
    }
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    fire.Fire(main)
