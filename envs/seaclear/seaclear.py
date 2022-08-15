"""SeaClear submarine environment module."""
import logging
from typing import Dict
from pathlib import Path

import numpy as np

import envs
from envs.flat_base import FlatBase
from envs.rotations import axisangle2quat, mat2embedding
from envs.utils import goal_distance

DEFAULT_MODEL_XML_PATH = str(Path("SeaClear", "seaclear.xml"))
FANCY_MODEL_XML_PATH = str(Path("SeaClear", "seaclear_fancy.xml"))

logger = logging.getLogger(__name__)


class SeaClear(FlatBase):
    """SeaClear submarine environment class."""

    gripper_type = "SeaClear"

    def __init__(self, fancy_world: bool = False):
        """Initialize a new flat environment.

        Args:
            object_name: Name of the manipulation object in Mujoco
            fancy_world: Loads a version with plants for visualization if True.
        """
        self.mocap_offset = np.array([0., 0., 0.])
        self._obs_threshold = 0.2  # Inflated threshold for training
        self._obs_check_threshold = 0.15  # Threshold used for success checks
        self._obs_range = 0.05  # Inner obstacle range for opposing goal and object positions
        self._obs_violation = False
        self._opposing_goal_object = False
        model_xml_path = FANCY_MODEL_XML_PATH if fancy_world else DEFAULT_MODEL_XML_PATH
        super().__init__(model_xml_path=model_xml_path,
                         gripper_extra_height=0.45,
                         initial_qpos={},
                         object_name="can",
                         n_actions=5,
                         object_size_range=0,
                         initial_gripper=[0.7, 0.7])
        self.height_offset = 0.0
        self.target_range = 0.3
        self.gripper_init_range = 0.25
        self.gripper_init_pos = np.array([0., 0., 0.45])
        self.current_object_pose = None
        # If true, use goal and object positions on opposite sides of the forbidden region

    def _set_action(self, action: np.ndarray):
        assert action.shape == (self.n_actions,)
        action = (action.copy())  # ensure that we don't change the action outside of this scope
        pos_ctrl, rot_ctrl, gripper_ctrl = action[:3], action[3], action[4]
        pos_ctrl *= 0.05  # limit maximum change in position
        pos_ctrl += self.mocap_offset
        rot_ctrl = axisangle2quat(0, 0, 1, np.pi * rot_ctrl)
        rot_ctrl *= 0.05  # limit maximum change in orientation
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        envs.utils.ctrl_set_action(self.sim, action)
        envs.utils.mocap_set_action(self.sim, action)

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

        # If the object is inside the obstacle, sample a goal where no collision occurs
        obs_pos = self.goal[3:6].copy()
        while goal_distance(obs_pos, object_pos) < self._obs_threshold:
            obs_pos[:2] = self.np_random.uniform(-self.target_range, self.target_range, size=2)
        achieved_goal = np.concatenate((object_pos, obs_pos))

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
            "achieved_goal": achieved_goal,
            "desired_goal": self.goal.copy(),
        }

    def compute_reward(self, achieved_goal: np.ndarray, goal: np.ndarray, _) -> np.ndarray:
        """Compute the agent reward for the achieved goal.

        Args:
            achieved_goal: Achieved goal.
            goal: Desired goal.
        """
        goal_d = goal_distance(achieved_goal[..., :3], goal[..., :3])
        obstacle_d = goal_distance(achieved_goal[..., :3], goal[..., 3:6])
        goal_reward = -(goal_d > self.target_threshold).astype(np.float32)
        obstacle_reward = -(obstacle_d < self._obs_threshold).astype(np.float32)
        return goal_reward + obstacle_reward

    def _render_callback(self):
        # Visualize target.
        site_id = self.sim.model.site_name2id("target0")
        self.sim.model.site_pos[site_id] = self.goal[:3]
        # Visualize obstacles
        site_id = self.sim.model.site_name2id("obstacle0")
        self.sim.model.site_pos[site_id] = self.goal[3:6]
        self.sim.forward()

    def _step_callback(self):
        object_pos = self.sim.data.get_site_xpos(self.object_name)
        goal_d = goal_distance(object_pos[:3], self.goal[3:6])
        self._obs_violation = goal_d < self._obs_check_threshold or self._obs_violation

    def _set_object_pose(self):
        object_pose = self.sim.data.get_joint_qpos(self.object_name + ":joint")
        if self._opposing_goal_object:
            # Calculate the object position in spherical coordinates around the env center
            object_angle = 2 * (self.np_random.rand() - 0.5) * np.pi
            object_range = self._obs_threshold + 0.05 + self.np_random.rand() * 0.05
            object_pose[:2] = np.array([np.cos(object_angle), np.sin(object_angle)]) * object_range
            object_pose[2] = 0.04
        else:
            object_pose[:2] = self.np_random.uniform(-self.target_range, self.target_range, size=2)
            object_pose[2] = 0.04
        self.current_object_pose = object_pose
        self.sim.data.set_joint_qpos(self.object_name + ":joint", object_pose)

    def _set_gripper_pose(self):
        gripper_pos = np.array([0., 0., 0 + self.gripper_extra_height])
        d_pos = self.np_random.uniform(-self.gripper_init_range, self.gripper_init_range, size=2)
        gripper_pos[:2] += d_pos  # Add random initial position change
        gripper_rot = np.array([1., 0., 0., 0.])
        d_rot = axisangle2quat(0, 0, 1, np.random.rand() - 0.5)
        gripper_rot += (d_rot / np.linalg.norm(d_rot))  # Add random initial rotation change
        gripper_rot /= np.linalg.norm(gripper_rot)  # Renormalize for quaternion
        self.sim.data.set_mocap_pos("robot0:mocap", gripper_pos)
        self.sim.data.set_mocap_quat("robot0:mocap", gripper_rot)

    def _reset_sim(self) -> bool:
        self.sim.set_state(self.initial_state)
        self._opposing_goal_object = self.np_random.rand() > 0.5
        self._env_setup(self.initial_qpos)
        self._obs_violation = False
        return True

    def _sample_goal(self) -> np.ndarray:
        object_pos = self.sim.data.get_site_xpos(self.object_name)
        goal = np.zeros(6)
        if self._opposing_goal_object:
            goal[:3] = -object_pos.copy()
            goal[2] = 0.05
            goal[3:5] = self.np_random.uniform(-self._obs_range, self._obs_range, size=2)
            goal[5] = 0.05
            d_obj = goal_distance(goal[3:6], object_pos[:3])
            d_goal = goal_distance(goal[3:6], goal[:3])
            d_grip = goal_distance(goal[3:6], self.gripper_start_pos[:6])
            while any([d < self._obs_threshold for d in (d_obj, d_goal, d_grip)]):
                goal[3:5] = self.np_random.uniform(-self._obs_range, self._obs_range, size=2)
                d_obj = goal_distance(goal[3:6], object_pos[:3])
                d_goal = goal_distance(goal[3:6], goal[:3])
                d_grip = goal_distance(goal[3:6], self.gripper_start_pos[:6])
        else:
            goal[:3] = self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal[2] = self.np_random.uniform(0.2, 0.4)
            goal[3:5] = self.np_random.uniform(-self.target_range, self.target_range, size=2)
            goal[5] = 0.05
            d_obj = goal_distance(goal[3:6], object_pos[:3])
            d_goal = goal_distance(goal[3:6], goal[:3])
            d_grip = goal_distance(goal[3:6], self.gripper_start_pos[:6])
            while any([d < self._obs_threshold for d in (d_obj, d_goal, d_grip)]):
                goal[3:5] = self.np_random.uniform(-self.target_range, self.target_range, size=2)
                d_obj = goal_distance(goal[3:6], object_pos[:3])
                d_goal = goal_distance(goal[3:6], goal[:3])
                d_grip = goal_distance(goal[3:6], self.gripper_start_pos[:6])
        self._goal = goal
        return goal

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        d = envs.utils.goal_distance(achieved_goal[:3], desired_goal[:3])
        success = np.logical_and(d < self.target_threshold, np.logical_not(self._obs_violation))
        return success.astype(np.float32)

    def _env_setup(self, initial_qpos: np.ndarray):
        self._modify_object_size()
        for name, value in initial_qpos.items():
            # Envs have joint start values for multiple grasp objects. Objects are not present in
            # all MuJoCo envs to speed up simulation for single grasp object environments. Therefore
            # non-existent joints are expected
            if name not in self.sim.model.joint_names:
                continue
            self.sim.data.set_joint_qpos(name, value)
        envs.utils.reset_mocap_welds(self.sim)
        self.sim.forward()
        # Move end effector into position
        self._set_gripper_pose()
        # Change object pose
        self._set_object_pose()
        # Run sim
        for _ in range(10):
            if self.initial_gripper:
                self.sim.data.ctrl[:] = self.initial_gripper
            self.sim.step()
        # Extract information for sampling goals
        self.gripper_start_pos = self.sim.data.get_site_xpos("robot0:grip").copy()
