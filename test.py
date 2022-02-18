"""Test a previously trained agent on an OpenAI gym environment."""

import argparse
import logging
from pathlib import Path
import time

import pickle
import torch
import gym
import mujoco_py

from mp_rl.core.utils import unwrap_obs
from mp_rl.core.actor import ActorNetwork


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
    parser.add_argument("--loglvl",
                        help="Logger levels",
                        choices=["DEBUG", "INFO", "WARN", "ERROR"],
                        default="INFO")
    parser.add_argument("--ntests", help="Number of evaluation runs", default=10, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    logger = logging.getLogger("GymTestScript")
    loglvls = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARN": logging.WARN,
        "ERROR": logging.ERROR
    }
    logging.basicConfig()
    logging.getLogger().setLevel(loglvls[args.loglvl])
    env = gym.make(args.env)
    size_s = len(env.observation_space["observation"].low) + len(
        env.observation_space["desired_goal"].low)
    size_a = len(env.action_space.low)
    actor = ActorNetwork(size_s, size_a)
    path = Path(__file__).parent / "mp_rl" / "saves" / args.env
    actor.load_state_dict(torch.load(path / "actor.pt"))
    with open(path / "state_norm.pkl", "rb") as f:
        state_norm = pickle.load(f)
    with open(path / "goal_norm.pkl", "rb") as f:
        goal_norm = pickle.load(f)
    success = 0.
    render = True
    for _ in range(args.ntests):
        state, goal, _ = unwrap_obs(env.reset())
        for _ in range(env._max_episode_steps):
            state, goal = state_norm(state), goal_norm(goal)
            state, goal = torch.as_tensor(state,
                                          dtype=torch.float32), torch.as_tensor(goal,
                                                                                dtype=torch.float32)
            with torch.no_grad():
                action = actor(torch.cat([state, goal]))
            next_obs, reward, _, info = env.step(action.numpy())
            state, goal, _ = unwrap_obs(next_obs)
            if render:
                try:
                    env.render()
                    time.sleep(0.1)
                except mujoco_py.cymj.GlfwError:
                    logger.warning("No display available, rendering disabled")
                    render = False
        success += info["is_success"]
    logger.info(f"Agent success rate: {success/args.ntests:.2f}")
