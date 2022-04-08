from typing import Dict, Optional
from pathlib import Path

import numpy as np
import json

import envs
from envs.flat_base import FlatBase

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
    "robot0:slide0": 0.4049,  # Robot arm
    "robot0:slide1": 0.48,
    "robot0:slide2": 0.0,
    "robot0:WRJ1": -0.16514339750464327,  # ShadowHand
    "robot0:WRJ0": -0.31973286565062153,
    "robot0:FFJ3": 0.14340512546557435,
    "robot0:FFJ2": 0.32028208333591573,
    "robot0:FFJ1": 0.7126053607727917,
    "robot0:FFJ0": 0.6705281001412586,
    "robot0:MFJ3": 0.000246444303701037,
    "robot0:MFJ2": 0.3152655251085491,
    "robot0:MFJ1": 0.7659800313729842,
    "robot0:MFJ0": 0.7323156897425923,
    "robot0:RFJ3": 0.00038520700007378114,
    "robot0:RFJ2": 0.36743546201985233,
    "robot0:RFJ1": 0.7119514095008576,
    "robot0:RFJ0": 0.6699446327514138,
    "robot0:LFJ4": 0.0525442258033891,
    "robot0:LFJ3": -0.13615534724474673,
    "robot0:LFJ2": 0.39872030433433003,
    "robot0:LFJ1": 0.7415570009679252,
    "robot0:LFJ0": 0.704096378652974,
    "robot0:THJ4": 0.003673823825070126,
    "robot0:THJ3": 0.5506291436028695,
    "robot0:THJ2": -0.014515151997119306,
    "robot0:THJ1": -0.0015229223564485414,
    "robot0:THJ0": -0.7894883021600622,
}


class FlatSHBase(FlatBase):

    EIGENGRASPS = EIGENGRASPS

    def __init__(self,
                 object_name: str,
                 model_xml_path: str,
                 n_eigengrasps: Optional[int] = None,
                 object_size_range: float = 0):
        """Initialize a new flat environment.

        Args:
            object_name: Name of the manipulation object in Mujoco
            model_xml_path: Mujoco world xml file path
            n_eigengrasps: Number of eigengrasps to use
        """
        self.n_eigengrasps = n_eigengrasps or 0
        assert 0 <= self.n_eigengrasps < 21, "Only [0, 20] eigengrasps available for the ShadowHand"
        n_actions = 7 + (n_eigengrasps or 20)
        super().__init__(model_xml_path=model_xml_path,
                         gripper_extra_height=0.3,
                         initial_qpos=DEFAULT_INITIAL_QPOS,
                         n_actions=n_actions,
                         object_name=object_name,
                         object_size_range=object_size_range)
        self._ctrl_range = self.sim.model.actuator_ctrlrange
        self._act_range = (self._ctrl_range[:, 1] - self._ctrl_range[:, 0]) / 2.0
        self._act_center = (self._ctrl_range[:, 1] + self._ctrl_range[:, 0]) / 2.0

    def _set_action(self, action):
        assert action.shape == (self.n_actions,)
        action = (action.copy())  # ensure that we don't change the action outside of this scope
        if self.n_eigengrasps:
            action = self._map_eigengrasps(action)
        assert action.shape == (27,)  # At this point, the action should always have full dimension
        pos_ctrl, rot_ctrl, hand_ctrl = action[:3], action[3:7], action[7:]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl /= np.linalg.norm(rot_ctrl)  # Norm quaternion
        rot_ctrl *= 0.05  # limit maximum change in orientation
        action = np.concatenate([pos_ctrl, rot_ctrl])

        # Apply action to simulation.
        envs.utils.mocap_set_action(self.sim, action)
        self.sim.data.ctrl[:] = self._act_center + hand_ctrl * self._act_range
        self.sim.data.ctrl[:] = np.clip(self.sim.data.ctrl, self._ctrl_range[:, 0],
                                        self._ctrl_range[:, 1])

    def _map_eigengrasps(self, action):
        pos_ctrl, hand_ctrl = action[:-self.n_eigengrasps], action[-self.n_eigengrasps:]
        # Transform hand controls to eigengrasps
        hand_ctrl = envs.utils.map_sh2mujoco(hand_ctrl @ self.EIGENGRASPS[:self.n_eigengrasps])
        np.clip(hand_ctrl, -1, 1, out=hand_ctrl)
        return np.concatenate((pos_ctrl, hand_ctrl))

    def _get_obs(self) -> Dict[str, np.ndarray]:
        # positions
        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip") * dt
        robot_qpos, robot_qvel = envs.utils.robot_get_obs(self.sim)
        object_pos = self.sim.data.get_site_xpos(self.object_name)
        # rotations
        object_rot = envs.rotations.mat2euler(self.sim.data.get_site_xmat(self.object_name))
        # velocities
        object_velp = self.sim.data.get_site_xvelp(self.object_name) * dt
        object_velr = self.sim.data.get_site_xvelr(self.object_name) * dt
        # gripper state
        object_rel_pos = object_pos - grip_pos
        object_velp -= grip_velp
        hand_state = robot_qpos[-24:]

        achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate([
            grip_pos,
            grip_velp,
            hand_state,
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
