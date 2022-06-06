"""FlatSH environment base module."""
import logging
from typing import Dict, Optional
from pathlib import Path

import numpy as np
import json

import envs
from envs.flat_base import FlatBase
from envs.rotations import mat2quat, mat2embedding

# The eigengrasps are exctracted from joint configurations obtained by fitting the ShadowHand to
# hand poses from the ContactPose dataset. For more information, see
# https://github.com/amacati/sh_eigen  TODO: Make repository public
with open(Path(__file__).parent / "eigengrasps.json", "r") as f:
    eigengrasps = json.load(f)
    assert all([len(value["joints"]) == 20 for value in eigengrasps.values()])
EIGENGRASPS = np.array([eigengrasps[str(i)]["joints"] for i in range(len(eigengrasps))])

DEFAULT_INITIAL_QPOS = {
    "cube:joint": [.1, -.1, .025, 1., 0, 0, 0],  # Grasp objects
    "cylinder:joint": [-.1, .1, .025, 1., 0, 0, 0],
    "sphere:joint": [.1, .1, .025, 1., 0, 0, 0],
    "mesh:joint": [-.1, -.1, .025, 1., 0, 0, 0],
    # Arm
    "panda_joint1": 1.5,
    "panda_joint2": 0.4,
    "panda_joint3": -1.5,
    "panda_joint4": -2.,
    "panda_joint5": 0,
    "panda_joint6": 2.8,
    "panda_joint7": 0,
    # ShadowHand
    "robot0:WRJ1": -0.1651,
    "robot0:WRJ0": -0.3197,
    "robot0:FFJ3": 0.1434,
    "robot0:FFJ2": 0.3202,
    "robot0:FFJ1": 0.7126,
    "robot0:FFJ0": 0.6705,
    "robot0:MFJ3": 0.0002,
    "robot0:MFJ2": 0.3152,
    "robot0:MFJ1": 0.7659,
    "robot0:MFJ0": 0.7323,
    "robot0:RFJ3": 0.0003,
    "robot0:RFJ2": 0.3674,
    "robot0:RFJ1": 0.7119,
    "robot0:RFJ0": 0.6699,
    "robot0:LFJ4": 0.0525,
    "robot0:LFJ3": -0.1361,
    "robot0:LFJ2": 0.3987,
    "robot0:LFJ1": 0.7415,
    "robot0:LFJ0": 0.7040,
    "robot0:THJ4": 0.0036,
    "robot0:THJ3": 0.5506,
    "robot0:THJ2": -0.0145,
    "robot0:THJ1": -0.0015,
    "robot0:THJ0": -0.7894,
}

DEFAULT_INITIAL_GRIPPER = [
    -0.1651, -0.3197, 0.1434, 0.3202, 0.7126, 0.0002, 0.3152, 0.7659, 0.0003, 0.3674, 0.7119,
    0.0525, -0.1361, 0.3987, 0.7415, 0.0036, 0.5506, -0.0145, -0.0015, -0.7894
]

logger = logging.getLogger(__name__)


class FlatSHBase(FlatBase):
    """FlatSH environment base class."""

    EIGENGRASPS = EIGENGRASPS
    gripper_type = "SH"

    def __init__(self,
                 object_name: str,
                 model_xml_path: str,
                 n_eigengrasps: Optional[int] = None,
                 object_size_multiplier: float = 1.,
                 object_size_range: float = 0.):
        """Initialize a flat ShadowHand environment.

        Args:
            object_name: Name of the manipulation object in Mujoco.
            model_xml_path: Mujoco world xml file path.
            n_eigengrasps: Number of eigengrasps to use.
            object_size_multiplier: Optional multiplier to change object sizes by a fixed amount.
            object_size_range: Optional range to randomly enlarge/shrink object sizes.
        """
        self.n_eigengrasps = n_eigengrasps or 0
        assert 0 <= self.n_eigengrasps < 21, "Only [0, 20] eigengrasps available for the ShadowHand"
        n_actions = 12 + (n_eigengrasps or 20)
        super().__init__(model_xml_path=model_xml_path,
                         gripper_extra_height=0.3,
                         initial_qpos=DEFAULT_INITIAL_QPOS,
                         initial_gripper=DEFAULT_INITIAL_GRIPPER,
                         n_actions=n_actions,
                         object_name=object_name,
                         object_size_multiplier=object_size_multiplier,
                         object_size_range=object_size_range)
        self._ctrl_range = self.sim.model.actuator_ctrlrange
        self._act_range = (self._ctrl_range[:, 1] - self._ctrl_range[:, 0]) / 2.0
        self._act_center = (self._ctrl_range[:, 1] + self._ctrl_range[:, 0]) / 2.0

    def _set_action(self, action: np.ndarray):
        assert action.shape == (self.n_actions,)
        action = (action.copy())  # ensure that we don't change the action outside of this scope
        if self.n_eigengrasps:
            action = self._map_eigengrasps(action)
        assert action.shape == (32,)  # At this point, the action should always have full dimension
        pos_ctrl, rot_ctrl, hand_ctrl = action[:3], action[3:12], action[12:]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = mat2quat(rot_ctrl.reshape(3, 3))
        rot_ctrl *= 0.05  # limit maximum change in orientation
        action = np.concatenate([pos_ctrl, rot_ctrl])

        # Apply action to simulation.
        envs.utils.mocap_set_action(self.sim, action)
        self.sim.data.ctrl[:] = self._act_center + hand_ctrl * self._act_range
        self.sim.data.ctrl[:] = np.clip(self.sim.data.ctrl, self._ctrl_range[:, 0],
                                        self._ctrl_range[:, 1])

    def _map_eigengrasps(self, action: np.ndarray) -> np.ndarray:
        pos_ctrl, hand_ctrl = action[:-self.n_eigengrasps], action[-self.n_eigengrasps:]
        # Transform hand controls to eigengrasps
        hand_ctrl = envs.utils.map_sh2mujoco(hand_ctrl @ self.EIGENGRASPS[:self.n_eigengrasps])
        np.clip(hand_ctrl, -1, 1, out=hand_ctrl)
        return np.concatenate((pos_ctrl, hand_ctrl))

    def _get_obs(self) -> Dict[str, np.ndarray]:
        # positions
        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        robot_qpos, _ = envs.utils.robot_get_obs(self.sim)
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
        grip_state = robot_qpos[-24:]

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
