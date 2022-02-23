"""OrientPickAndPlace class file."""
from pathlib import Path
from gym import utils
import numpy as np
from envs.fetch import FetchEnv
import envs.utils

MODEL_XML_PATH = str(Path("fetch", "pick_and_place.xml"))


class OrientPickAndPlace(FetchEnv, utils.EzPickle):
    """Environment for pick and place with orientation control enabled."""

    def __init__(self, reward_type: str = "sparse"):
        """Initialize the Mujoco sim."""
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        FetchEnv.__init__(self,
                          MODEL_XML_PATH,
                          has_object=True,
                          block_gripper=False,
                          n_substeps=20,
                          gripper_extra_height=0.2,
                          target_in_the_air=True,
                          target_offset=0.0,
                          obj_range=0.15,
                          target_range=0.15,
                          distance_threshold=0.05,
                          initial_qpos=initial_qpos,
                          reward_type=reward_type,
                          n_actions=8)
        utils.EzPickle.__init__(self, reward_type=reward_type)

    def _set_action(self, action: np.ndarray):
        assert action.shape == (8,)
        action = (action.copy())  # ensure that we don't change the action outside of this scope
        pose_ctrl, gripper_ctrl = action[:7], action[7]

        pose_ctrl[:3] *= 0.05  # limit maximum change in position
        pose_ctrl[3:7] *= 0.05  # Limit maximum change in rotation
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pose_ctrl, gripper_ctrl])
        # Apply action to simulation.
        envs.utils.ctrl_set_action(self.sim, action)
        # mocap_set_action with quaternion is implemented as summing up vectors, does NOT account
        # for quaternion specialness. TODO: Investigate if this needs to change
        envs.utils.mocap_set_action(self.sim, action)
