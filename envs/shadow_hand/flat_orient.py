"""FlatSHOrient environment module."""
from pathlib import Path
from typing import Any, Dict, Optional

from gym import utils
import numpy as np

import envs
from envs.shadow_hand.flat_base import FlatSHBase
from envs.rotations import embedding2mat, embedding2quat, quat2embedding, mat2embedding, quat_mul
from envs.rotations import axisangle2quat

MODEL_XML_PATH = str(Path("ShadowHand", "flat_orient.xml"))


class FlatSHOrient(FlatSHBase, utils.EzPickle):
    """FlatSHOrient environment class."""

    def __init__(self,
                 n_eigengrasps: Optional[int] = None,
                 object_size_multiplier: float = 1.,
                 object_size_range: float = 0.,
                 angle_reduce_factor: float = 1.25,
                 angle_min_tolerance: float = 0.2,
                 angle_reduce_performance: float = 0.75):
        """Initialize a shadow hand mesh environment with additional orientation goals.

        Args:
            n_eigengrasps: Number of eigengrasps to use.
            object_size_multiplier: Optional multiplier to change object sizes by a fixed amount.
            object_size_range: Optional range to randomly enlarge/shrink object sizes.
        """
        self.curr_object_rot = np.array([1, 0, 0, 0])
        FlatSHBase.__init__(self,
                            object_name="mesh",
                            model_xml_path=MODEL_XML_PATH,
                            n_eigengrasps=n_eigengrasps,
                            object_size_multiplier=object_size_multiplier,
                            object_size_range=object_size_range)
        utils.EzPickle.__init__(self,
                                n_eigengrasps=n_eigengrasps,
                                object_size_multiplier=object_size_multiplier,
                                object_size_range=object_size_range,
                                angle_reduce_factor=angle_reduce_factor,
                                angle_min_tolerance=angle_min_tolerance,
                                angle_reduce_performance=angle_reduce_performance)
        self.angle_threshold = np.pi
        self.angle_reduce_factor = angle_reduce_factor
        self.angle_min_tolerance = angle_min_tolerance
        self.angle_reduce_performance = angle_reduce_performance
        self.early_stop_ok = False

    def _env_setup(self, initial_qpos: np.ndarray):
        # Mesh falls if it is rotated around its y axis -> reposition on setup so that the object
        # settles during the initial 10 steps.
        object_pose = self.sim.data.get_joint_qpos(self.object_name + ":joint")
        object_pose[:2] = self.sim.data.get_body_xpos("table0")[:2]
        object_pose[2] = self.height_offset
        # Rotate object
        object_rot = axisangle2quat(0, 0, 1, (self.np_random.rand() - 0.5) * np.pi)
        object_pose[3:7] = object_rot
        self.sim.data.set_joint_qpos(self.object_name + ":joint", object_pose)
        # Remove mesh from initial_qpos to keep the modified joint position
        modified_qpos = initial_qpos.copy()
        modified_qpos.pop("mesh:joint", None)
        super()._env_setup(modified_qpos)

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

        achieved_goal = np.concatenate([object_pos, object_rot])

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

    def _sample_goal(self) -> np.ndarray:
        table_pos = self.sim.data.get_body_xpos("table0")[:3]
        object_pos = self.sim.data.get_site_xpos(self.object_name)[:3]
        goal = np.zeros(9)
        goal[:2] = table_pos[:2]
        goal[:2] += self.np_random.uniform(-self.target_range, self.target_range, size=2)
        while np.linalg.norm(object_pos[:2] - goal[:2]) < 0.1:
            goal[:2] = table_pos[:2]
            goal[:2] += self.np_random.uniform(-self.target_range, self.target_range, size=2)
        goal[2] = self.height_offset
        if self.np_random.uniform() < 0.5:
            goal[2] += self.np_random.uniform(0, self.goal_max_height)
        q_1 = axisangle2quat(0, 0, 1, (self.np_random.rand() - 0.5) * np.pi)
        q_2 = axisangle2quat(1, 0, 0, (self.np_random.rand() - 0.5) * np.pi / 5)
        q_3 = axisangle2quat(0, 1, 0, (self.np_random.rand() - 0.5) * np.pi / 5)
        q = quat_mul(q_1, q_2)
        q = quat_mul(q, q_3)
        goal[3:9] = quat2embedding(q)
        return goal.copy()

    def compute_reward(self, achieved_goal: np.ndarray, goal: np.ndarray, _) -> np.ndarray:
        """Compute the agent reward for the achieved goal.

        Args:
            achieved_goal: Achieved goal.
            goal: Desired goal.
        """
        # Compute distance between goal and the achieved goal.
        d = envs.utils.goal_distance(achieved_goal[..., :3], goal[..., :3])
        if self.angle_threshold == np.pi:  # Speed up reward calculation for initial training
            return -(d > self.target_threshold).astype(np.float32)
        qgoal = embedding2quat(goal[..., 3:9])
        qachieved_goal = embedding2quat(achieved_goal[..., 3:9])
        angle = 2 * np.arccos(np.clip(np.abs(np.sum(qachieved_goal * qgoal, axis=-1)), -1, 1))
        assert (angle < np.pi + 1e-3).all(), "Angle greater than pi encountered in quat difference"
        pos_error = d > self.target_threshold
        rot_error = angle > self.angle_threshold
        return -(np.logical_or(pos_error, rot_error)).astype(np.float32)

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        d = envs.utils.goal_distance(achieved_goal[..., :3], desired_goal[..., :3])
        qgoal = embedding2quat(desired_goal[..., 3:9])
        qachieved_goal = embedding2quat(achieved_goal[..., 3:9])
        angle = 2 * np.arccos(np.clip(np.abs(np.sum(qachieved_goal * qgoal, axis=-1)), -1, 1))
        assert (angle < np.pi + 1e-3).all(), "Angle greater than pi encountered in quat difference"
        pos_success = d < self.target_threshold
        rot_success = angle < self.angle_threshold
        return (np.logical_and(pos_success, rot_success)).astype(np.float32)

    def _render_callback(self):
        # Visualize target.
        mat = embedding2mat(self.goal[3:9])
        quat = embedding2quat(self.goal[3:9])
        site_id = self.sim.model.site_name2id("target0")
        self.sim.model.site_pos[site_id] = self.goal[:3]
        self.sim.model.site_quat[site_id] = quat
        site_id = self.sim.model.site_name2id("target0x")
        dx = mat @ np.array([0.05, 0, 0])
        self.sim.model.site_pos[site_id] = self.goal[:3] + dx
        self.sim.model.site_quat[site_id] = quat
        site_id = self.sim.model.site_name2id("target0y")
        dy = mat @ np.array([0, 0.05, 0])
        self.sim.model.site_pos[site_id] = self.goal[:3] + dy
        self.sim.model.site_quat[site_id] = quat
        site_id = self.sim.model.site_name2id("target0z")
        dz = mat @ np.array([0, 0, 0.05])
        self.sim.model.site_pos[site_id] = self.goal[:3] + dz
        self.sim.model.site_quat[site_id] = quat
        self.sim.forward()

    def epoch_callback(self, _: Any, av_success: float):
        """Reduce the angle threshold depending on the agent performance.

        Args:
            av_success: Current average agent success.
        """
        if av_success > self.angle_reduce_performance:
            angle_reduced_tolerance = self.angle_threshold / self.angle_reduce_factor
            self.angle_threshold = max(angle_reduced_tolerance, self.angle_min_tolerance)
        if self.angle_threshold == self.angle_min_tolerance:
            self.early_stop_ok = True
