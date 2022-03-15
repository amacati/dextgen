"""ShadowHandMultiObject class file."""
from pathlib import Path
from typing import Dict

from gym import utils
import numpy as np
import json

import envs.robot_env
import envs.utils
import envs.rotations
from envs.shadow_hand.shadowhand_base import ShadowHandBase, DEFAULT_INITIAL_QPOS

MODEL_XML_PATH = str(Path("shfetch", "shadowhand_multiobject.xml"))

# The eigengrasps are exctracted from joint configurations obtained by fitting the ShadowHand to
# hand poses from the ContactPose dataset. For more information, see
# https://github.com/amacati/sh_eigen  TODO: Make repository public
with open(Path(__file__).parent / "eigengrasps.json", "r") as f:
    eigengrasps = json.load(f)
    assert all([len(value["joints"]) == 20 for value in eigengrasps.values()])
EIGENGRASPS = np.array([eigengrasps[str(i)]["joints"] for i in range(len(eigengrasps))])


class ShadowHandMultiObject(ShadowHandBase, utils.EzPickle):
    """Environment for pick and place with the ShadowHand and eigengrasps."""

    EIGENGRASPS = EIGENGRASPS

    def __init__(self,
                 reward_type: str = "sparse",
                 n_eigengrasps: int = 1,
                 p_grasp_start: float = 0.):
        """Initialize the Mujoco sim.

        Params:
            reward_type: Choice of reward formular.
            n_eigengrasps: Number of eigengrasp vectors the agent gets as action input.
            p_grasp_start: Fraction of episode starts with pregrasped objects.
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
        self.curr_obj_id = 0  # Necessary for initial observation call
        assert n_eigengrasps <= 20
        self.n_eigengrasps = n_eigengrasps
        self.p_grasp_start = p_grasp_start
        n_actions = 3 + n_eigengrasps
        initial_qpos = DEFAULT_INITIAL_QPOS.copy()
        initial_qpos["object1:joint"] = [1.35, 0.63, 0.42, 1., 0, 0, 0]
        initial_qpos["object2:joint"] = [1.45, 0.53, 0.4, 1., 0, 0, 0]
        super().__init__(n_actions=n_actions,
                         reward_type=reward_type,
                         p_grasp_start=p_grasp_start,
                         model_path=MODEL_XML_PATH,
                         initial_qpos=initial_qpos)
        utils.EzPickle.__init__(self,
                                reward_type=reward_type,
                                n_eigengrasps=n_eigengrasps,
                                p_grasp_start=p_grasp_start)

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

    def _reset_sim(self) -> bool:
        if np.random.rand() < self.p_grasp_start:
            return self._reset_sim_grasp()
        self.sim.set_state(self.initial_state)
        # Randomize start position of object.
        object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
            -self.obj_range, self.obj_range, size=2)
        while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
            object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                -self.obj_range, self.obj_range, size=2)
        self.curr_obj_id = np.random.choice(3)
        for i in range(3):
            curr_obj_joint = "object" + str(i) + ":joint"
            if self.curr_obj_id == i:
                object_qpos = self.sim.data.get_joint_qpos(curr_obj_joint)
                assert object_qpos.shape == (7,)
                object_qpos[:2] = object_xpos
                self.sim.data.set_joint_qpos(curr_obj_joint, object_qpos)
            else:
                object_qpos = self.sim.data.get_joint_qpos(curr_obj_joint)
                assert object_qpos.shape == (7,)
                object_qpos[:2] = i * 0.1  # Move objects to sim origin without causing collision
                self.sim.data.set_joint_qpos(curr_obj_joint, object_qpos)
        self.sim.forward()
        return True

    def _reset_sim_grasp(self) -> bool:
        self.sim.set_state(self.initial_state_grasp)
        for i in range(1, 3):
            curr_obj_joint = "object" + str(i) + ":joint"
            object_qpos = self.sim.data.get_joint_qpos(curr_obj_joint)
            assert object_qpos.shape == (7,)
            object_qpos[:2] = i * 0.1  # Move objects to sim origin without causing collision
            self.sim.data.set_joint_qpos(curr_obj_joint, object_qpos)
        self.sim.forward()
        return True

    def _get_obs(self) -> Dict[str, np.ndarray]:
        curr_obj = "object" + str(self.curr_obj_id)
        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip") * dt
        robot_qpos, robot_qvel = envs.utils.robot_get_obs(self.sim)
        object_pos = self.sim.data.get_site_xpos(curr_obj)
        # rotations
        object_rot = envs.rotations.mat2euler(self.sim.data.get_site_xmat(curr_obj))
        # velocities
        object_velp = self.sim.data.get_site_xvelp(curr_obj) * dt
        object_velr = self.sim.data.get_site_xvelr(curr_obj) * dt
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
            "desired_goal": self.goal.copy()
        }
