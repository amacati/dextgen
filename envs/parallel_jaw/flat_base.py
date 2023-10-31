"""FlatPJ environment base module."""
import logging
from typing import Dict, Tuple

import numpy as np

import envs
from envs.flat_base import FlatBase
from envs.rotations import mat2embedding

logger = logging.getLogger(__name__)


class FlatPJBase(FlatBase):
    """FlatPJ environment base class."""

    gripper_type = "ParallelJaw"

    def __init__(self,
                 model_xml_path: str,
                 p_high_goal: float,
                 goal_range: Tuple[float, float],
                 n_actions: int = 4):
        """Initialize a flat parallel jaw environment.

        Args:
            model_xml_path: Path to the Mujoco xml world file.
            n_actions: Action dimensionality of the environment.
        """
        initial_qpos = {
            "panda_joint1": 0,
            "panda_joint2": 0.4,
            "panda_joint3": 0.,
            "panda_joint4": -2.4,
            "panda_joint5": 0,
            "panda_joint6": 2.8,
            "panda_joint7": 0,
            "cube:joint": [.1, -.1, .025, 1., 0, 0, 0],
            "cylinder:joint": [-.1, .1, .025, 1., 0, 0, 0],
            "sphere:joint": [.1, .1, .025, 1., 0, 0, 0],
            "mesh:joint": [-.1, -.1, .025, 1., 0, 0, 0]
        }
        super().__init__(model_xml_path=model_xml_path,
                         gripper_extra_height=0.15,
                         initial_qpos=initial_qpos,
                         p_high_goal=p_high_goal,
                         goal_range=goal_range,
                         n_actions=n_actions)
        self._ctrl_range = self.sim.model.actuator_ctrlrange
        self._act_range = (self._ctrl_range[:, 1] - self._ctrl_range[:, 0]) / 2.0
        self._act_center = (self._ctrl_range[:, 1] + self._ctrl_range[:, 0]) / 2.0

    def _set_action(self, action: np.ndarray):
        assert action.shape == (self.n_actions,)
        action = (action.copy())  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [
            1.0,
            0.0,
            1.0,
            0.0,
        ]
        pose_ctrl = np.concatenate([pos_ctrl, rot_ctrl])
        # Apply action to simulation.
        self.sim.data.ctrl[:] = self._act_center + gripper_ctrl * self._act_range
        # envs.utils.ctrl_set_action(self.sim, action)
        envs.utils.mocap_set_action(self.sim, pose_ctrl)

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

        achieved_goal = object_pos

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
