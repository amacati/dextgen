"""ShadowHandOrientation class file."""
from typing import Optional
import copy
import time

from gym import utils
import numpy as np

import envs
from envs.shadow_hand.shadowhand_base import ShadowHandBase


class ShadowHandOrientation(ShadowHandBase, utils.EzPickle):
    """Environment for pick and place with the ShadowHand and free hand orientation."""

    def __init__(self,
                 reward_type: str = "sparse",
                 n_eigengrasps: Optional[int] = None,
                 p_rotation_start: float = 0.,
                 p_grasp_start: float = 0.):
        """Initialize the Mujoco sim.

        Params:
            reward_type: Choice of reward formular.
            n_eigengrasps: Number of eigengrasp vectors the agent gets as action input.
            p_rotation_start: Fraction of episodes which start with different arm orientation.
            p_grasp_start: Fraction of episodes which start with pregrasped objects. Note that this
                applies to non rotation starts only, so the actual probability is
                (1 - `p_rotation_start`) * `p_grasp_start`.
        """
        self.p_rotation_start = p_rotation_start
        n_actions = 7 + (n_eigengrasps or 20)
        super().__init__(n_actions=n_actions,
                         reward_type=reward_type,
                         p_grasp_start=p_grasp_start,
                         n_eigengrasps=n_eigengrasps)
        utils.EzPickle.__init__(self,
                                reward_type=reward_type,
                                n_eigengrasps=n_eigengrasps,
                                p_rotation_start=p_rotation_start,
                                p_grasp_start=p_grasp_start)

    def _set_default_action(self, action: np.ndarray):
        assert action.shape == (27,)
        action = action.copy()
        move_ctrl, hand_ctrl = action[:7], action[7:]
        # Apply action to simulation.
        envs.utils.mocap_set_action(self.sim, move_ctrl * 0.05)  # Limit maximum change in position
        self.sim.data.ctrl[:] = self._act_center + hand_ctrl * self._act_range
        self.sim.data.ctrl[:] = np.clip(self.sim.data.ctrl, self._ctrl_range[:, 0],
                                        self._ctrl_range[:, 1])

    def _env_setup(self, initial_qpos: np.ndarray):
        if self.p_grasp_start > 0:
            self._env_setup_grasp(initial_qpos)
            self.initial_state_grasp = copy.deepcopy(self.sim.get_state())
        if self.p_rotation_start > 0:
            self._env_setup_rotation(initial_qpos)
            self.initial_state_rotation = copy.deepcopy(self.sim.get_state())
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        envs.utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([1.32, 0.75, 0.355 + self.gripper_extra_height])
        gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        self.sim.data.set_mocap_pos("robot0:mocap", gripper_target)
        self.sim.data.set_mocap_quat("robot0:mocap", gripper_rotation)
        ctrl_range = self.sim.model.actuator_ctrlrange
        act_range = (ctrl_range[:, 1] - ctrl_range[:, 0]) / 2.0
        act_center = (ctrl_range[:, 1] + ctrl_range[:, 0]) / 2.0
        hand_ctrl = np.array(
            [1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1, -1, 0, -1, -1, 0, 1, -1, -1, 0])
        self.sim.data.ctrl[:] = act_center + hand_ctrl * act_range
        for _ in range(10):
            self.sim.step()
        self.initial_gripper_xpos = self.sim.data.get_site_xpos("robot0:grip").copy()
        self.height_offset = self.sim.data.get_site_xpos("object0")[2]

    def _env_setup_rotation(self, initial_qpos: np.ndarray):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        envs.utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([1.3, 0.65, 0.255 + self.gripper_extra_height])
        gripper_rotation = [0.5, 0.5, 0.5, 0.5]
        self.sim.data.set_mocap_pos("robot0:mocap", gripper_target)
        self.sim.data.set_mocap_quat("robot0:mocap", gripper_rotation)
        ctrl_range = self.sim.model.actuator_ctrlrange
        act_range = (ctrl_range[:, 1] - ctrl_range[:, 0]) / 2.0
        act_center = (ctrl_range[:, 1] + ctrl_range[:, 0]) / 2.0
        hand_ctrl = np.array(
            [1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1, -1, 0, -1, -1, 0, 1, -1, -1, 0])
        self.sim.data.ctrl[:] = act_center + hand_ctrl * act_range
        for _ in range(10):
            self.sim.step()

    def _reset_sim(self) -> bool:
        if np.random.rand() > self.p_rotation_start:
            return super()._reset_sim()
        self.sim.set_state(self.initial_state_rotation)
        # Randomize start position of object.
        object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
            -self.obj_range, self.obj_range, size=2)
        while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
            object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                -self.obj_range, self.obj_range, size=2)
        object_qpos = self.sim.data.get_joint_qpos("object0:joint")
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        self.sim.data.set_joint_qpos("object0:joint", object_qpos)
        self.sim.forward()
        return True
