from pathlib import Path
import logging
import pickle
import time

import torch
import gym
import mujoco_py
import envs  # Import registers environments with gym  # noqa: F401

from mp_rl.core.actor import PosePolicyNet
from mp_rl.core.utils import unwrap_obs
from parse_args import parse_args

from optim.utils import check_grasp
from optim.control import Controller

logger = logging.getLogger(__name__)


def main():
    args = parse_args()
    assert args.env == "FlatPJCube-v0", "Only FlatPJCube-v0 supported for optimization"
    logger = logging.getLogger("OptimTestScript")
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    env = gym.make(args.env)
    size_g = len(env.observation_space["desired_goal"].low)
    size_s = len(env.observation_space["observation"].low) + size_g
    size_a = len(env.action_space.low)
    actor = PosePolicyNet(size_s, size_a, args.actor_net_nlayers, args.actor_net_layer_width)
    path = Path(__file__).parent.parent / "saves" / args.env
    actor.load_state_dict(torch.load(path / "actor.pt"))
    with open(path / "state_norm.pkl", "rb") as f:
        state_norm = pickle.load(f)
    with open(path / "goal_norm.pkl", "rb") as f:
        goal_norm = pickle.load(f)
    success = 0.
    render = args.render == "y"
    env.use_contact_info()
    env.save_reset()
    controller = Controller()

    for i in range(args.ntests):
        state, goal, _ = unwrap_obs(env.reset())
        done = False
        early_stop = 0
        while not done:
            state, goal = state_norm(state), goal_norm(goal)
            state = torch.as_tensor(state, dtype=torch.float32)
            goal = torch.as_tensor(goal, dtype=torch.float32)
            with torch.no_grad():
                action = actor(torch.cat([state, goal])).numpy()
            next_obs, reward, done, info = env.step(action)
            state, goal, _ = unwrap_obs(next_obs)
            if render:
                try:
                    env.render()
                    time.sleep(0.04)  # Gym operates on 25 Hz
                except mujoco_py.cymj.GlfwError:
                    logger.warning("No display available, rendering disabled")
                    render = False
            if check_grasp(info):
                info["gripper_info"]["next_state"] = 0.
                break
        if not check_grasp(info):
            logger.warning("Failed to generate grasp proposal. Skipping trial")
            continue
        state, goal, _ = unwrap_obs(env.load_reset())
        controller.reset()
        logger.info("Failed to generate grasp proposal. Skipping trial")
        controller.optimize_grasp(info)
        done = False
        while not done:
            next_obs, reward, done, info = env.step(controller(state, goal))
            state, goal, _ = unwrap_obs(next_obs)
            early_stop = (early_stop + 1) if not reward else 0
            if early_stop == 10:
                break
        success += info["is_success"]
    logger.info(f"Agent success rate: {success/args.ntests:.2f}")


if __name__ == "__main__":
    main()
