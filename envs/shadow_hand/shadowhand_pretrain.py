"""ShadowHandEigengrasps class file."""
from typing import Optional
from pathlib import Path

from gym import utils
import numpy as np

import envs
from envs.shadow_hand.shadowhand_base import ShadowHandBase

MODEL_XML_PATH = str(Path("shfetch", "shadowhand_pretrain.xml"))


class ShadowHandPretrain(ShadowHandBase, utils.EzPickle):
    """Environment for pick and place with the ShadowHand and eigengrasps."""

    def __init__(self,
                 reward_type: str = "sparse",
                 n_eigengrasps: Optional[int] = None,
                 p_grasp_start: float = 0.):
        """Initialize the Mujoco sim.

        Params:
            reward_type: Choice of reward formular.
            n_eigengrasps: Number of eigengrasp vectors the agent gets as action input.
            p_grasp_start: Fraction of episode starts with pregrasped objects.
        """
        n_actions = 3 + (n_eigengrasps or 20)
        super().__init__(n_actions=n_actions,
                         reward_type=reward_type,
                         p_grasp_start=p_grasp_start,
                         model_path=MODEL_XML_PATH,
                         n_eigengrasps=n_eigengrasps)
        utils.EzPickle.__init__(self,
                                reward_type=reward_type,
                                n_eigengrasps=n_eigengrasps,
                                p_grasp_start=p_grasp_start)

    def _env_setup_grasp(self, initial_qpos: np.ndarray):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        envs.utils.reset_mocap_welds(self.sim)
        self.sim.forward()
        # Move end effector into position and configuration.
        hand_ctrl = np.array(
            [1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1, -1, 0, -1, -1, 0, 1, -1, -1, 0])
        ctrl_range = self.sim.model.actuator_ctrlrange
        act_range = (ctrl_range[:, 1] - ctrl_range[:, 0]) / 2.0
        act_center = (ctrl_range[:, 1] + ctrl_range[:, 0]) / 2.0
        self.sim.data.ctrl[:] = act_center + hand_ctrl * act_range
        gripper_target = np.array([1.32, 0.75, 0.355 + self.gripper_extra_height])
        gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        self.sim.data.set_mocap_pos("robot0:mocap", gripper_target)
        self.sim.data.set_mocap_quat("robot0:mocap", gripper_rotation)
        for _ in range(10):
            self.sim.step()
        # Set object position below gripper
        object_xpos = self.sim.data.get_site_xpos("robot0:grip")[:2] + np.array([0.052, 0.02])
        object_qpos = self.sim.data.get_joint_qpos("object0:joint")
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        self.sim.data.set_joint_qpos("object0:joint", object_qpos)
        self.sim.forward()
        # Predefined grasp sequence
        for i in range(30):
            if i < 10:
                action = np.array([0, 0, -0.3])
                hand_ctrl = np.array(
                    [1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1, -1, 0, -1, -1, 0, 1, -1, -1, 0])
            elif i < 20:
                action = np.array([0, 0, 0])
                hand_ctrl = -np.array(
                    [1, 1, 0, -.5, 0, 0, 0, 0, 0, -.5, -.5, -.5, 0, -.5, 1, 0, -1, 1, 1, 0])
            else:
                action = np.array([0, 0, 0.5])
                hand_ctrl = -np.array(
                    [1, 1, 0, -.5, 0, 0, 0, 0, 0, -.5, -.5, -.5, 0, -.5, 1, 0, -1, 1, 1, 0])
            action = np.concatenate((action * 0.05, np.array([1., 0., 1., 0.])))
            envs.utils.mocap_set_action(self.sim, action)
            self.sim.data.ctrl[:] = act_center + hand_ctrl * act_range
            self.sim.data.ctrl[:] = np.clip(self.sim.data.ctrl, ctrl_range[:, 0], ctrl_range[:, 1])
            self.sim.step()
