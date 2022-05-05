"""FlatPJOrient environment module."""
from pathlib import Path
from typing import Dict

from gym import utils
import numpy as np

import envs
from envs.parallel_jaw.flat_base import FlatPJBase
from envs.rotations import embedding2quat, quat2embedding, mat2embedding

MODEL_XML_PATH = str(Path("pj", "flat_pj_orient.xml"))


class FlatPJOrient(FlatPJBase, utils.EzPickle):
    """FlatPJOrient environment class."""

    def __init__(self, object_size_range: float = 0):
        """Initialize a parallel jaw cube environment with additional orientation goals.

        Args:
            object_size_range: Optional range to enlarge/shrink object sizes.
        """
        FlatPJBase.__init__(self,
                            object_name="cube",
                            model_xml_path=MODEL_XML_PATH,
                            object_size_range=object_size_range)
        utils.EzPickle.__init__(self, object_size_range=object_size_range)
        self.angle_threshold = np.pi * 0.1

    def _sample_object_pose(self) -> np.ndarray:
        object_pose = super()._sample_object_pose()
        # Random rotation around z axis
        object_pose[3] = self.np_random.rand()
        object_pose[6] = self.np_random.rand()
        object_pose[3:7] /= np.linalg.norm(object_pose[3:7])
        return object_pose

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
        quat = np.zeros(4)
        quat[0] = self.np_random.rand()
        quat[3] = self.np_random.rand()
        quat /= np.linalg.norm(quat)
        goal[3:9] = quat2embedding(quat)
        return goal.copy()

    def compute_reward(self, achieved_goal: np.ndarray, goal: np.ndarray, _) -> np.ndarray:
        """Compute the agent reward for the achieved goal.

        Args:
            achieved_goal: Achieved goal.
            goal: Desired goal.
        """
        # Compute distance between goal and the achieved goal.
        d = envs.utils.goal_distance(achieved_goal[..., :3], goal[..., :3])
        qgoal = embedding2quat(goal[..., 3:9])
        qachieved_goal = embedding2quat(achieved_goal[..., 3:9])
        angle = 2 * np.arccos(np.clip(np.abs(np.sum(qachieved_goal * qgoal, axis=-1)), -1, 1))
        assert (angle < np.pi + 1e-3).all(), "Angle greater than pi encountered in quat difference"
        pos_error = d > self.target_threshold
        rot_error = angle > self.angle_threshold
        return -(np.logical_or(pos_error, rot_error)).astype(np.float32)

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        d = envs.utils.goal_distance(achieved_goal[:3], desired_goal[:3])
        return (d < self.target_threshold).astype(np.float32)

    def _render_callback(self):
        # Visualize target.
        body_id = self.sim.model.body_name2id("target0")
        self.sim.model.body_pos[body_id] = self.goal[:3]
        self.sim.model.body_quat[body_id] = embedding2quat(self.goal[3:9])
        self.sim.forward()
