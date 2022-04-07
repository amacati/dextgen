from typing import Dict
from pathlib import Path

import numpy as np

import envs
from envs.flat_base import FlatBase

MODEL_XML_PATH = str(Path("pj", "pick_and_place.xml"))


class FlatPJBase(FlatBase):

    def __init__(self, obj_name: str, obj_size_range: float = 0):
        """Initialize a new flat environment.

        Args:
            obj_name: Name of the manipulation object in Mujoco
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
        super().__init__(model_xml_path=MODEL_XML_PATH,
                         gripper_extra_height=0.2,
                         target_offset=0.,
                         initial_qpos=initial_qpos,
                         n_actions=8,
                         obj_name=obj_name,
                         obj_size_range=obj_size_range)

    def _set_action(self, action):
        assert action.shape == (self.n_actions,)
        action = (action.copy())  # ensure that we don't change the action outside of this scope
        pos_ctrl, rot_ctrl, gripper_ctrl = action[:3], action[3:7], action[7]

        pos_ctrl *= 0.05  # limit maximum change in position
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
        object_pos = self.sim.data.get_site_xpos(self.obj_name)
        # rotations
        object_rot = envs.rotations.mat2euler(self.sim.data.get_site_xmat(self.obj_name))
        # velocities
        object_velp = self.sim.data.get_site_xvelp(self.obj_name) * dt
        object_velr = self.sim.data.get_site_xvelr(self.obj_name) * dt
        # gripper state
        object_rel_pos = object_pos - grip_pos
        object_velp -= grip_velp
        gripper_state = robot_qpos[-2:]
        gripper_vel = (robot_qvel[-2:] * dt)  # change to a scalar if the gripper is made symmetric

        achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate([
            grip_pos,
            object_pos.ravel(),
            object_rel_pos.ravel(),
            gripper_state,
            object_rot.ravel(),
            object_velp.ravel(),
            object_velr.ravel(),
            grip_velp,
            gripper_vel,
        ])

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }
