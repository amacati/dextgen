"""Test script for randomly sampled optimization initializations."""
import logging

import numpy as np
import gym
import envs  # noqa: F401  environments need to be registered with the gym module
from envs.rotations import embedding2mat  # Import registers environments with gym  # noqa: F401

from mp_rl.core.utils import unwrap_obs
from optim.control import Controller
from optim.utils.rotations import quat2mat
from parse_args import parse_args

logger = logging.getLogger(__name__)


def main():
    """Reset the simulation, sample a random initialization and run the optimization."""
    args = parse_args()
    assert args.env == "FlatPJCube-v0", "Only FlatPJCube-v0 supported for optimization"
    logger = logging.getLogger("OptimTestScript")
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    env = gym.make(args.env)
    env.seed(0)
    np_random = np.random.default_rng(1)
    hasconverged = 0
    controller = Controller()

    for i in range(args.ntests):
        # Sample initial environment state, extract object pose information
        state, _, _ = unwrap_obs(env.reset())
        # sample random configuration around the cube
        xinit = np.zeros(9)
        xinit[:3] = state[14:17] + np_random.uniform(-0.1, 0.1, 3)
        xinit[3:7] = np_random.uniform(-1, 1, 4)
        xinit[3:7] /= np.linalg.norm(xinit[3:7])
        xinit[8] = np_random.uniform(-1, 1)

        # Optimize proposed grasp
        # Create info consistent with pose initialization. Contact forces, positions etc are not
        # used, therefore can remain empty
        info = {
            "gripper_info": {
                "pos": xinit[:3],
                "orient": quat2mat(xinit[3:7]),
                "state": [xinit[7], xinit[7]],
                "next_state": -1,
                "type": "ParallelJaw"
            },
            "contact_info": [{
                "geom1": "robot0:r_gripper_finger_link",
                "contact_force": [],
                "pos": [],
                "frame": []
            }, {
                "geom1": "robot0:l_gripper_finger_link",
                "contact_force": [],
                "pos": [],
                "frame": []
            }],
            "object_info": {
                "pos": state[14:17],
                "orient": embedding2mat(state[20:26]),
                "size": [0.025, 0.025, 0.025],
                "name": "cube"
            }
        }

        controller.reset()
        try:
            controller.optimize_grasp(info)
            hasconverged += 1
        except Exception as e:  # noqa: E722
            logger.warning(e)  # Sample failed optimization, don't increase convergence counter
    logger.info(f"Number of tests: {args.ntests}, Converged: {hasconverged}")


if __name__ == "__main__":
    main()
