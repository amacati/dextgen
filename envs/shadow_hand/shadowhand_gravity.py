"""ShadowHandGravity class file."""
from pathlib import Path

from gym import utils
import numpy as np
import json

import envs.robot_env
import envs.utils
import envs.rotations
from envs.shadow_hand.shadowhand_base import ShadowHandBase

# The eigengrasps are exctracted from joint configurations obtained by fitting the ShadowHand to
# hand poses from the ContactPose dataset. For more information, see
# https://github.com/amacati/sh_eigen  TODO: Make repository public
with open(Path(__file__).parent / "eigengrasps.json", "r") as f:
    eigengrasps = json.load(f)
    assert all([len(value["joints"]) == 20 for value in eigengrasps.values()])
EIGENGRASPS = np.array([eigengrasps[str(i)]["joints"] for i in range(len(eigengrasps))])


class ShadowHandGravity(ShadowHandBase, utils.EzPickle):
    """Environment for pick and place with the ShadowHand and eigengrasps."""

    EIGENGRASPS = EIGENGRASPS

    def __init__(self,
                 reward_type: str = "sparse",
                 n_eigengrasps: int = 1,
                 n_increase_epochs: int = 100):
        """Initialize the Mujoco sim.

        Params:
            reward_type: Choice of reward formular.
            n_eigengrasps: Number of eigengrasp vectors the agent gets as action input.
        """
        self.c_low = (1.05, 0.4, 0.4)
        self.c_high = (1.55, 1.1, 0.4)
        self.max_reset_steps = 100
        self.distance_threshold = 0.05
        self.target_in_the_air = True
        self.target_range = 0.15
        self.target_offset = 0.0
        self.gripper_extra_height = 0.35
        self.reward_type = reward_type
        self.obj_range = 0.15
        assert n_eigengrasps <= 20
        self.n_eigengrasps = n_eigengrasps
        self.n_increase_epochs = n_increase_epochs
        n_actions = 3 + n_eigengrasps
        super().__init__(n_actions=n_actions, reward_type=reward_type, p_grasp_start=0)
        utils.EzPickle.__init__(self,
                                reward_type=reward_type,
                                n_eigengrasps=n_eigengrasps,
                                n_increase_epochs=n_increase_epochs)
        self.sim.model.opt.gravity[-1] = 0

    def _set_action(self, action: np.ndarray):
        """Map the action vector to eigengrasps and write the resulting action to Mujoco.

        Params:
            Action: Action value vector.
        """
        assert action.shape == (3 + self.n_eigengrasps,)
        action = (action.copy())  # ensure that we don't change the action outside of this scope
        pos_ctrl, hand_ctrl = action[:3], action[3:]
        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1.0, 0.0, 1.0, 0.0]  # fixed rotation of the end effector as a quaternion
        action = np.concatenate([pos_ctrl, rot_ctrl])
        # Transform hand controls to eigengrasps
        hand_ctrl = envs.utils.map_sh2mujoco(hand_ctrl @ self.EIGENGRASPS[:self.n_eigengrasps])
        np.clip(hand_ctrl, -1, 1, out=hand_ctrl)

        # Apply action to simulation.
        envs.utils.mocap_set_action(self.sim, action)
        self.sim.data.ctrl[:] = self._act_center + hand_ctrl * self._act_range
        self.sim.data.ctrl[:] = np.clip(self.sim.data.ctrl, self._ctrl_range[:, 0],
                                        self._ctrl_range[:, 1])

    def epoch_callback(self, epoch: int):
        """Increase environment gravity stepwise after each epoch.

        Args:
            epoch: Current training epoch.
            max_epoch: Maximum number of epochs.
        """
        self.sim.model.opt.gravity[-1] = max(min(1, epoch / (self.n_increase_epochs)), 0) * -9.81

    def _reset_sim(self) -> bool:
        self.sim.set_state(self.initial_state)
        # Randomize start position of object.
        object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
            -self.obj_range, self.obj_range, size=2)
        while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
            object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                -self.obj_range, self.obj_range, size=2)
        object_qpos = self.sim.data.get_joint_qpos("object0:joint")
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        # Stop random placement in the air when gravity approaches real value
        if np.random.rand() > 0.5 and self.sim.model.opt.gravity[-1] < 5:
            object_qpos[2] += np.random.rand() * 0.2
        self.sim.data.set_joint_qpos("object0:joint", object_qpos)
        self.sim.forward()
        return True
