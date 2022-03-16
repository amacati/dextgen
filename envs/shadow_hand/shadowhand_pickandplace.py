"""ShadowHandPickAndPlace class file."""
from gym import utils
import numpy as np

import envs.robot_env
import envs.utils
import envs.rotations
from envs.shadow_hand.shadowhand_base import ShadowHandBase


class ShadowHandPickAndPlace(ShadowHandBase, utils.EzPickle):
    """Environment for pick and place with the ShadowHand."""

    def __init__(self, reward_type: str = "sparse", p_grasp_start: float = 0.):
        """Initialize the Mujoco sim.

        Params:
            reward_type: Choice of reward formular.
            p_grasp_start: Fraction of episode starts with pregrasped objects.
        """
        super().__init__(n_actions=23, reward_type=reward_type, p_grasp_start=p_grasp_start)
        utils.EzPickle.__init__(self, reward_type=reward_type, p_grasp_start=p_grasp_start)

    def _set_action(self, action: np.ndarray):
        """Map the action vector to the robot and dwrite the resulting action to Mujoco.

        Params:
            Action: Action value vector.
        """
        assert action.shape == (23,)
        action = (action.copy())  # ensure that we don't change the action outside of this scope
        pos_ctrl, hand_ctrl = action[:3], action[3:]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1.0, 0.0, 1.0, 0.0]  # fixed rotation of the end effector as a quaternion
        action = np.concatenate([pos_ctrl, rot_ctrl])

        # Apply action to simulation.
        envs.utils.mocap_set_action(self.sim, action)
        self.sim.data.ctrl[:] = self._act_center + hand_ctrl * self._act_range
        self.sim.data.ctrl[:] = np.clip(self.sim.data.ctrl, self._ctrl_range[:, 0],
                                        self._ctrl_range[:, 1])
