"""FlatPJ environment base module."""
import logging
from typing import Dict

import numpy as np

import envs
from envs.flat_base import FlatBase
from envs.rotations import mat2quat, mat2embedding

logger = logging.getLogger(__name__)


class FlatPJBase(FlatBase):
    """FlatPJ environment base class."""

    gripper_type = "PJ"

    def __init__(self, object_name: str, model_xml_path: str, object_size_range: float = 0):
        """Initialize a flat parallel jaw environment.

        Args:
            object_name: Name of the manipulation object in Mujoco.
            model_xml_path: Path to the Mujoco xml world file.
            object_size_range: Range of object size modification. If 0, modification is disabled.
        """
        initial_qpos = {
            "robot0:slide0": 0.4049,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "cube:joint": [.1, -.1, .025, 1., 0, 0, 0],
            "cylinder:joint": [-.1, .1, .025, 1., 0, 0, 0],
            "sphere:joint": [.1, .1, .025, 1., 0, 0, 0],
            "mesh:joint": [-.1, -.1, .025, 1., 0, 0, 0]
        }
        super().__init__(
            model_xml_path=model_xml_path,
            gripper_extra_height=0.2,
            initial_qpos=initial_qpos,
            n_actions=13,  # 3 pos, 9 rot, 1 gripper
            object_name=object_name,
            object_size_range=object_size_range)

    def _set_action(self, action: np.ndarray):
        assert action.shape == (self.n_actions,)
        action = (action.copy())  # ensure that we don't change the action outside of this scope
        pos_ctrl, rot_ctrl, gripper_ctrl = action[:3], action[3:12], action[12]

        pos_ctrl *= 0.05  # limit maximum change in position
        # Transform rot_ctrl from matrix to quaternion
        rot_ctrl = mat2quat(rot_ctrl.reshape(3, 3))
        rot_ctrl *= 0.05  # limit maximum change in orientation
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        envs.utils.ctrl_set_action(self.sim, action)
        envs.utils.mocap_set_action(self.sim, action)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        # positions
        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        robot_qpos, robot_qvel = envs.utils.robot_get_obs(self.sim)
        object_pos = self.sim.data.get_site_xpos(self.object_name)
        object_rel_pos = object_pos - grip_pos
        # rotations
        grip_rot_mat = self.sim.data.get_site_xmat("robot0:grip")
        object_rot_mat = self.sim.data.get_site_xmat(self.object_name)
        grip_rot = mat2embedding(grip_rot_mat)
        object_rot = mat2embedding(object_rot_mat)
        object_rel_rot = mat2embedding(grip_rot_mat.T @ object_rot_mat)
        # velocities
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip") * dt
        object_velp = self.sim.data.get_site_xvelp(self.object_name) * dt
        object_velr = self.sim.data.get_site_xvelr(self.object_name) * dt
        object_velp -= grip_velp
        # gripper state
        grip_state = robot_qpos[-2:]

        achieved_goal = np.squeeze(object_pos.copy())

        obs = np.concatenate([
            grip_pos,
            grip_rot,
            grip_state,
            grip_velp,
            object_pos,
            object_rel_pos,
            object_rot,
            object_rel_rot,
            object_velp,
            object_velr,
        ])

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }
