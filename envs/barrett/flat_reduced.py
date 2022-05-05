"""FlatBarrettReduced environment module."""
import logging
from typing import Dict, Optional
from pathlib import Path

import numpy as np
import json

import envs
from envs.flat_base import FlatBase
from envs.rotations import mat2embedding

# Eigengrasps for barrett are designed by hand
with open(Path(__file__).parent / "eigengrasps.json", "r") as f:
    eigengrasps = json.load(f)
    assert all([len(value["joints"]) == 4 for value in eigengrasps.values()])
EIGENGRASPS = np.array([eigengrasps[str(i)]["joints"] for i in range(len(eigengrasps))])

DEFAULT_INITIAL_QPOS = {
    "cube:joint": [.1, -.1, .025, 1., 0, 0, 0],  # Grasp objects
    "cylinder:joint": [-.1, .1, .025, 1., 0, 0, 0],
    "sphere:joint": [.1, .1, .025, 1., 0, 0, 0],
    "mesh:joint": [-.1, -.1, .025, 1., 0, 0, 0],
    "robot0:slide0": 0.4049,  # Robot arm
    "robot0:slide1": 0.48,
    "robot0:slide2": 0.0,
    "robot0:bhand_finger1_prox_joint": 1.5708,
    "robot0:bhand_finger2_prox_joint": 1.5708,
    "robot0:bhand_finger1_med_joint": 1.22173,
    "robot0:bhand_finger2_med_joint": 1.22173,
    "robot0:bhand_finger3_med_joint": 1.22173,
    "robot0:bhand_finger1_dist_joint": 0.40724,
    "robot0:bhand_finger2_dist_joint": 0.40724,
    "robot0:bhand_finger3_dist_joint": 0.40724,
}

DEFAULT_INITIAL_GRIPPER = [1.5708, 1.22173, 1.22173, 1.22173]

MODEL_XML_PATH = str(Path("barrett", "flat_barrett_cube.xml"))

logger = logging.getLogger(__name__)


class FlatBarrettReduced(FlatBase):
    """FlatBarrettReduced environment class."""

    EIGENGRASPS = EIGENGRASPS
    gripper_type = "Barrett"

    def __init__(self, n_eigengrasps: Optional[int] = None, object_size_range: float = 0):
        """Initialize a flat BarrettHand cube environment without orientation controls.

        Args:
            n_eigengrasps: Number of eigengrasps to use
            object_size_range: Optional range to enlarge/shrink object sizes.
        """
        self.n_eigengrasps = n_eigengrasps or 0
        assert 0 <= self.n_eigengrasps < 5, "Only [0, 4] eigengrasps available for the Barrett hand"
        n_actions = 3 + (n_eigengrasps or 4)
        super().__init__(model_xml_path=MODEL_XML_PATH,
                         gripper_extra_height=0.3,
                         initial_qpos=DEFAULT_INITIAL_QPOS,
                         initial_gripper=DEFAULT_INITIAL_GRIPPER,
                         n_actions=n_actions,
                         object_name="cube",
                         object_size_range=object_size_range)
        self._ctrl_range = self.sim.model.actuator_ctrlrange
        self._act_range = (self._ctrl_range[:, 1] - self._ctrl_range[:, 0]) / 2.0
        self._act_center = (self._ctrl_range[:, 1] + self._ctrl_range[:, 0]) / 2.0

    def _set_action(self, action: np.ndarray):
        assert action.shape == (self.n_actions,)
        action = (action.copy())  # ensure that we don't change the action outside of this scope
        if self.n_eigengrasps:
            action = self._map_eigengrasps(action)
        assert action.shape == (7,)  # At this point, the action should always have full dimension
        pos_ctrl, hand_ctrl = action[:3], action[3:]
        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = np.array([1.0, 0.0, 1.0, 0.0])
        action = np.concatenate([pos_ctrl, rot_ctrl])

        # Apply action to simulation.
        envs.utils.mocap_set_action(self.sim, action)
        self.sim.data.ctrl[:] = self._act_center + hand_ctrl * self._act_range
        self.sim.data.ctrl[:] = np.clip(self.sim.data.ctrl, self._ctrl_range[:, 0],
                                        self._ctrl_range[:, 1])

    def _map_eigengrasps(self, action: np.ndarray) -> np.ndarray:
        pos_ctrl, hand_ctrl = action[:-self.n_eigengrasps], action[-self.n_eigengrasps:]
        # Transform hand controls to eigengrasps
        hand_ctrl = hand_ctrl @ self.EIGENGRASPS[:self.n_eigengrasps]
        np.clip(hand_ctrl, -1, 1, out=hand_ctrl)
        return np.concatenate((pos_ctrl, hand_ctrl))

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
        grip_state = robot_qpos[-8:]  # 2 prox, 3 med, 3 dist joints

        achieved_goal = np.squeeze(object_pos.copy())

        obs = np.concatenate([
            grip_pos,
            grip_rot,
            grip_state,
            grip_velp,
            object_pos.ravel(),
            object_rel_pos.ravel(),
            object_rot.ravel(),
            object_rel_rot,
            object_velp.ravel(),
            object_velr.ravel(),
        ])
        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }

    def _sample_object_pose(self) -> np.ndarray:
        object_pose = super()._sample_object_pose()
        object_pose[3:7] = [1, 0, 1, 0]
        return object_pose
