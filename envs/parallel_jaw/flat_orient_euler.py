"""FlatPJOrientEuler environment module."""
from pathlib import Path
from typing import Any, Dict

from gym import utils
import numpy as np

import envs
from envs.parallel_jaw.flat_base import FlatPJBase
from envs.rotations import embedding2quat, euler2quat, mat2embedding, quat2embedding, quat2mat

MODEL_XML_PATH = str(Path("PJ", "flat_orient.xml"))


class FlatPJOrientEuler(FlatPJBase, utils.EzPickle):
    """FlatPJOrientEuler environment class."""

    def __init__(self,
                 object_size_multiplier: float = 1.,
                 object_size_range: float = 0.,
                 angle_reduce_factor: float = 1.25,
                 angle_min_tolerance: float = 0.2,
                 angle_reduce_performance: float = 0.75):
        """Initialize a parallel jaw cube environment with additional euler orientation goals.

        Args:
            object_size_multiplier: Optional multiplier to change object sizes by a fixed amount.
            object_size_range: Optional range to randomly enlarge/shrink object sizes.
            angle_reduce_factor: Reduction factor per epoch callback.
            angle_min_tolerance: Minimum angle goal tolerance.
            angle_reduce_performance: Performance threshold above which an epoch callback reduces
                the angle tolerances.
        """
        self.curr_object_rot = np.array([1, 0, 0, 0])
        FlatPJBase.__init__(self,
                            object_name="cube",
                            model_xml_path=MODEL_XML_PATH,
                            object_size_multiplier=object_size_multiplier,
                            object_size_range=object_size_range,
                            n_actions=7)
        utils.EzPickle.__init__(self,
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

    def _sample_object_pose(self) -> np.ndarray:
        object_pose = super()._sample_object_pose()
        # Random rotation around z axis
        theta = (np.random.rand() - 0.5) * 0.1 + np.pi / 2
        euler = np.array([np.pi / 2, theta, np.pi / 2])
        object_rot = euler2quat(euler)
        self.curr_object_rot = object_rot
        object_pose[3:7] = object_rot
        return object_pose

    def _set_action(self, action: np.ndarray):
        assert action.shape == (self.n_actions,)
        action = (action.copy())  # ensure that we don't change the action outside of this scope
        pos_ctrl, rot_ctrl, gripper_ctrl = action[:3], action[3:6], action[6]

        pos_ctrl *= 0.05  # limit maximum change in position
        # Transform rot_ctrl from euler to quaternion
        rot_ctrl *= np.array([1., 1., 0.5]) * np.pi  # a and b in [-pi, pi], g in [-pi/2, pi/2]
        rot_ctrl = euler2quat(rot_ctrl)
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
        theta = (np.random.rand() - 0.5) * 3 + np.pi / 2
        euler = np.array([np.pi / 2, theta, np.pi / 2])
        goal[3:9] = quat2embedding(euler2quat(euler))
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
        quat = embedding2quat(self.goal[3:9])
        mat = quat2mat(quat)
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
