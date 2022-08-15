"""Optimization test script with learned initializations."""
from pathlib import Path
import logging
import pickle
import time
from typing import Dict

import torch
import numpy as np
import gym
import mujoco_py
import envs  # Import registers environments with gym  # noqa: F401

from mp_rl.core.actor import PosePolicyNet
from mp_rl.core.utils import unwrap_obs
from optim.grippers import get_gripper
from parse_args import parse_args
from optim.utils.utils import check_grasp
from optim.control import Controller

logger = logging.getLogger(__name__)


def compute_com_dist_improvement(info: Dict, xopt: np.ndarray) -> float:
    """Compute the improvement in distance of both fingers to the CoM.

    Args:
        info: Contact information dictionary.
        xopt: Optimized gripper configuration.

    Returns:
        The summed distance difference between the initial and optimal configuration of the fingers.
    """
    gripper = get_gripper(info)
    kin_fr = gripper.create_kinematics(gripper.LINKS[0], None)
    kin_fl = gripper.create_kinematics(gripper.LINKS[1], None)
    d1_init = np.linalg.norm(info["object_info"]["pos"] - kin_fr(gripper.state))
    d2_init = np.linalg.norm(info["object_info"]["pos"] - kin_fl(gripper.state))
    d1_opt = np.linalg.norm(info["object_info"]["pos"] - kin_fr(xopt))
    d2_opt = np.linalg.norm(info["object_info"]["pos"] - kin_fl(xopt))
    return d1_init + d2_init - d1_opt - d2_opt


def main():
    """Test the optimization scheme with learned priors.

    Loads the most recently trained agent from the saves dictionary, simulates the environment until
    a valid grasp has been found, resets the environment to the episode start, optimizes the gripper
    configuration and then tries to grasp the object with a controller.

    Note:
        Controller is not optimized, optimized grasps are not always achieved.
    """
    args = parse_args()
    assert args.env == "FlatPJCube-v0", "Only FlatPJCube-v0 supported for optimization"
    logger = logging.getLogger("OptimTestScript")
    logging.basicConfig()
    logging.getLogger().setLevel(args.loglvl)

    env = gym.make(args.env)
    env.seed(0)
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
    controller = Controller()
    hasconverged = 0
    com_dist_improvement = 0
    for i in range(args.ntests):
        # Generate grasp proposal. If the agent fails, reset and try again
        while True:
            state, goal, _ = unwrap_obs(env.reset())
            env.save_reset()
            done = False
            while not done:
                state, goal = state_norm(state), goal_norm(goal)
                state = torch.as_tensor(state, dtype=torch.float32)
                goal = torch.as_tensor(goal, dtype=torch.float32)
                with torch.no_grad():
                    action = actor(torch.cat([state, goal])).numpy()
                next_obs, _, done, info = env.step(action)
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
                    logger.info("Successful grasp detected, optimizing grasp pose")
                    # input("press enter to continue")
                    break
            if check_grasp(info):
                break
            logger.warning("Failed to generate grasp proposal. Retrying with another env")

        # Optimize proposed grasp
        env.reset()
        state, goal, _ = unwrap_obs(env.load_reset())
        if render:
            env.render()
        env.enable_full_orient_ctrl()
        controller.reset()
        try:
            logger.info("Trying to optimize grasp")
            xopt = controller.optimize_grasp(info)
            hasconverged += 1
            done = False
            early_stop = 0
            while not done:
                next_obs, reward, done, _info = env.step(controller(state, goal))
                if render:
                    env.render()
                state, goal, _ = unwrap_obs(next_obs)
                early_stop = (early_stop + 1) if not reward else 0
                if early_stop == 10:
                    break
            logger.info("Optimized control finished")
            success += _info["is_success"]
            com_dist_improvement += compute_com_dist_improvement(info, xopt)
        except RuntimeError as e:
            logger.warning(e)
        env.enable_full_orient_ctrl(False)
    logger.info(f"Agent success rate: {success/args.ntests:.2f}, converged: {hasconverged}")
    logger.info(f"Average distance to CoM improvement: {com_dist_improvement/hasconverged}")


if __name__ == "__main__":
    main()
