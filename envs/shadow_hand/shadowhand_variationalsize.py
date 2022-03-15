"""ShadowHandVariationalSize class file."""
from gym import utils
import numpy as np

import envs.robot_env
import envs.utils
import envs.rotations
from envs.shadow_hand.shadowhand_base import ShadowHandBase


class ShadowHandVariationalSize(ShadowHandBase, utils.EzPickle):
    """Environment for pick and place with the ShadowHand and variational cube size."""

    def __init__(self, reward_type: str = "sparse", p_grasp_start: float = 0.):
        """Initialize the Mujoco sim.

        Params:
            reward_type: Choice of reward formular.
            p_grasp_start: Fraction of episode starts with pregrasped objects.
        """
        self.c_low = (1.05, 0.4, 0.4)
        self.c_high = (1.55, 1.1, 0.4)
        self.cube_size = np.array([0.025, 0.025, 0.04])
        self.cube_deviation = 0.01
        self.max_reset_steps = 100
        self.distance_threshold = 0.05
        self.target_in_the_air = True
        self.target_range = 0.15
        self.target_offset = 0.0
        self.gripper_extra_height = 0.35
        self.reward_type = reward_type
        self.obj_range = 0.15
        self.p_grasp_start = p_grasp_start
        super().__init__(n_actions=23, reward_type=reward_type, p_grasp_start=p_grasp_start)
        utils.EzPickle.__init__(self, reward_type=reward_type, p_grasp_start=p_grasp_start)

    def _set_action(self, action: np.ndarray):
        """Map the action vector to the robot and dwrite the resulting action to Mujoco.

        Params:
            Action: Action value vector.
        """
        assert action.shape == (23,)
        action = (action.copy())  # ensure that we don't change the action outside of this scope
        pos_ctrl, hand_ctrl = action[:3], action[3:]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1.0, 0.0, 1.0, 0.0]  # fixed rotation of the end effector as a quaternion
        action = np.concatenate([pos_ctrl, rot_ctrl])

        # Apply action to simulation.
        envs.utils.mocap_set_action(self.sim, action)
        self.sim.data.ctrl[:] = self._act_center + hand_ctrl * self._act_range
        self.sim.data.ctrl[:] = np.clip(self.sim.data.ctrl, self._ctrl_range[:, 0],
                                        self._ctrl_range[:, 1])

    def _reset_sim(self) -> bool:
        if np.random.rand() < self.p_grasp_start:
            # Reset object0 size to normal
            self.sim.model.geom_size[-1] = self.cube_size
            return self._reset_sim_grasp()
        # Set object0 size to random
        cube_pertubation = 2 * (np.random.rand(3) - 0.5) * self.cube_deviation
        self.sim.model.geom_size[-1] = self.cube_size + cube_pertubation
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
        object_qpos[2] += cube_pertubation[2]
        self.sim.data.set_joint_qpos("object0:joint", object_qpos)
        self.sim.forward()
        return True
