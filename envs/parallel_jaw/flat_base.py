import logging
from typing import Dict

import numpy as np

import envs
from envs.flat_base import FlatBase

logger = logging.getLogger(__name__)


class FlatPJBase(FlatBase):

    def __init__(self, object_name: str, model_xml_path: str, object_size_range: float = 0):
        """Initialize a new flat environment.

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
        super().__init__(model_xml_path=model_xml_path,
                         gripper_extra_height=0.2,
                         initial_qpos=initial_qpos,
                         n_actions=8,
                         object_name=object_name,
                         object_size_range=object_size_range)

    def _set_action(self, action):
        assert action.shape == (self.n_actions,)
        action = (action.copy())  # ensure that we don't change the action outside of this scope
        pos_ctrl, rot_ctrl, gripper_ctrl = action[:3], action[3:7], action[7]

        pos_ctrl *= 0.05  # limit maximum change in position
        if np.linalg.norm(rot_ctrl) == 0:
            logger.warning("Zero quaternion encountered, setting to robot gripper orientation")
            rot_ctrl = self.sim.data.get_body_xquat("robot0:gripper_link")
        rot_ctrl /= np.linalg.norm(rot_ctrl)  # Norm quaternion
        rot_ctrl *= 0.05  # limit maximum change in orientation
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        envs.utils.ctrl_set_action(self.sim, action)
        envs.utils.mocap_set_action(self.sim, action)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        # positions
        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip") * dt
        robot_qpos, robot_qvel = envs.utils.robot_get_obs(self.sim)
        object_pos = self.sim.data.get_site_xpos(self.object_name)
        # rotations
        grip_rot = envs.rotations.mat2euler(self.sim.data.get_site_xmat("robot0:grip"))
        object_rot = envs.rotations.mat2euler(self.sim.data.get_site_xmat(self.object_name))
        # velocities
        object_velp = self.sim.data.get_site_xvelp(self.object_name) * dt
        object_velr = self.sim.data.get_site_xvelr(self.object_name) * dt
        # gripper state
        object_rel_pos = object_pos - grip_pos
        object_velp -= grip_velp
        gripper_state = robot_qpos[-2:]
        gripper_vel = (robot_qvel[-2:] * dt)  # change to a scalar if the gripper is made symmetric

        achieved_goal = np.squeeze(object_pos.copy())

        obs = np.concatenate([
            grip_pos,
            grip_rot,
            gripper_state,
            grip_velp,
            gripper_vel,  # TODO: Possibly remove
            object_pos.ravel(),
            object_rel_pos.ravel(),
            object_rot.ravel(),
            object_velp.ravel(),
            object_velr.ravel(),
        ])

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }
