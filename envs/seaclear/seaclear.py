"""SeaClear submarine environment module."""
from typing import Dict
from pathlib import Path

import numpy as np

import envs
from envs.flat_base import FlatBase
from envs.rotations import axisangle2quat, mat2embedding
from envs.utils import goal_distance

DEFAULT_MODEL_XML_PATH = str(Path("SeaClear", "seaclear.xml"))
FANCY_MODEL_XML_PATH = str(Path("SeaClear", "seaclear_fancy.xml"))


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
        self._obs_dist = 0.15
        self._obs_threshold = 0.1
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
            "achieved_goal": np.concatenate((achieved_goal, self.goal[3:6])),
            "desired_goal": self.goal.copy(),
        }

    def compute_reward(self, achieved_goal: np.ndarray, goal: np.ndarray, _) -> np.ndarray:
        """Compute the agent reward for the achieved goal.

        Args:
            achieved_goal: Achieved goal.
            goal: Desired goal.
        """
        goal_d = goal_distance(achieved_goal[..., :3], goal[..., :3])
        obstacle_d = goal_distance(achieved_goal[..., 3:6], goal[..., 3:6])
        goal_reward = -(goal_d > self.target_threshold).astype(np.float32)
        obstacle_reward = -(obstacle_d > self._obs_threshold).astype(np.float32)
        return goal_reward + obstacle_reward

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id("target0")
        self.sim.model.site_pos[site_id] = self.goal[:3] - sites_offset[0]
        # Visualize obstacle
        site_id = self.sim.model.site_name2id("obstacle0")
        self.sim.model.site_pos[site_id] = self.goal[3:6] - sites_offset[site_id]
        self.sim.forward()

    def _set_object_pose(self):
        object_pose = self.sim.data.get_joint_qpos(self.object_name + ":joint")
        object_pose[:2] = self.np_random.uniform(-self.target_range, self.target_range, size=2)
        object_pose[2] = 0.04
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

    def _sample_object_pose(self) -> np.ndarray:
        object_pose = np.zeros(7)
        object_pose[:2] = self.np_random.uniform(-self.target_range, self.target_range, size=2)
        return object_pose

    def _reset_sim(self) -> bool:
        self.sim.set_state(self.initial_state)
        self._env_setup(self.initial_qpos)  # Rerun env setup to get new start poses for the robot
        return True

    def _sample_goal(self) -> np.ndarray:
        goal = self.np_random.uniform(-self.target_range, self.target_range, size=3)
        goal[2] = self.np_random.uniform(0.2, 0.45)
        obstacle = self.np_random.uniform(-self.target_range, self.target_range, size=3)
        obstacle[2] = 1000  # TODO: Reenable obstacle
        object_pos = self.sim.data.get_site_xpos(self.object_name)
        d_obj = goal_distance(obstacle[:2], object_pos[:2])
        d_goal = goal_distance(obstacle[:2], goal[:2])
        d_grip = goal_distance(obstacle[:2], self.gripper_start_pos[:2])
        while any([d < self._obs_dist for d in (d_obj, d_goal, d_grip)]):
            obstacle = self.np_random.uniform(-self.target_range, self.target_range, size=3)
            obstacle[2] = 1000
            d_obj = goal_distance(obstacle[:2], object_pos[:2])
            d_goal = goal_distance(obstacle[:2], goal[:2])
            d_grip = goal_distance(obstacle[:2], self.gripper_start_pos[:2])
        return np.concatenate((goal, obstacle))

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        d = envs.utils.goal_distance(achieved_goal[:3], desired_goal[:3])
        return (d < self.target_threshold).astype(np.float32)

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
        # Save start positions on first run
        if self.gripper_init_pos is None:
            self.gripper_init_pos = np.array([0., 0., 0.45])
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
