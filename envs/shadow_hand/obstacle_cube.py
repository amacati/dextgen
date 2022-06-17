"""ObstacleSHCube environment module."""
from typing import Optional, Dict
from pathlib import Path

import numpy as np
from gym import utils

from envs.utils import goal_distance
from envs.shadow_hand.flat_base import FlatSHBase

MODEL_XML_PATH = str(Path("ShadowHand", "obstacle_cube.xml"))


class ObstacleSHCube(FlatSHBase, utils.EzPickle):
    """ObstacleSHCube environment class."""

    def __init__(self, n_eigengrasps: Optional[int] = None):
        """Initialize a ShadowHand cube environment with an obstacle to avoid.

        Args:
            n_eigengrasps: Number of eigengrasps to use
        """
        self.obstacle_threshold = 0.1
        super().__init__(object_name="cube",
                         model_xml_path=MODEL_XML_PATH,
                         n_eigengrasps=n_eigengrasps)
        utils.EzPickle.__init__(self, n_eigengrasps=n_eigengrasps)

    def _env_setup(self, initial_qpos: np.ndarray):
        initial_qpos = initial_qpos.copy()
        for key in ("cylinder:joint", "sphere:joint", "mesh:joint"):
            initial_qpos.pop(key, None)  # Delete all object joint defaults that are not in the XML
        super()._env_setup(initial_qpos)

    def _sample_object_pose(self) -> np.ndarray:
        object_pose = super()._sample_object_pose()
        # Random rotation around z axis
        object_pose[3] = self.np_random.rand()
        object_pose[6] = self.np_random.rand()
        object_pose[3:7] /= np.linalg.norm(object_pose[3:7])
        return object_pose

    def compute_reward(self, achieved_goal: np.ndarray, goal: np.ndarray, _) -> np.ndarray:
        """Compute the agent reward for the achieved goal.

        Args:
            achieved_goal: Achieved goal.
            goal: Desired goal.
        """
        # Compute distance between goal and the achieved goal.
        goal_d = goal_distance(achieved_goal[..., :3], goal[..., :3])
        obstacle_d = goal_distance(achieved_goal[..., 3:6], goal[..., 3:6])
        goal_reward = -(goal_d > self.target_threshold).astype(np.float32)
        obstacle_reward = -(obstacle_d > self.obstacle_threshold).astype(np.float32)
        return goal_reward + obstacle_reward

    def _sample_goal(self) -> np.ndarray:
        table_pos = self.sim.data.get_body_xpos("table0")[:3]
        object_pos = self.sim.data.get_site_xpos(self.object_name)[:3]
        goal = table_pos.copy()
        goal[:2] += self.np_random.uniform(-self.target_range, self.target_range, size=2)
        while np.linalg.norm(object_pos[:2] - goal[:2]) < 0.1:
            goal = table_pos[:3].copy()
            goal[:2] += self.np_random.uniform(-self.target_range, self.target_range, size=2)
            print(goal, object_pos)
        goal[2] = self.height_offset
        if self.np_random.uniform() < 0.5:
            goal[2] += self.np_random.uniform(0, self.goal_max_height)
        # Sample obstacle outside of hand and obstacle
        obstacle = table_pos.copy()
        obstacle[:2] += self.np_random.uniform(-self.target_range, self.target_range, size=2)
        obstacle[2] = self.height_offset
        while goal_distance(obstacle, goal) < 0.15 or goal_distance(
                self.gripper_start_pos, obstacle) < 0.15 or goal_distance(obstacle,
                                                                          object_pos) < 0.15:
            obstacle = table_pos.copy()
            obstacle[:2] += self.np_random.uniform(-self.target_range, self.target_range, size=2)
            obstacle[2] = self.height_offset
        return np.concatenate((goal, obstacle))

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        d = goal_distance(achieved_goal[:3], desired_goal[:3])
        return (d < self.target_threshold).astype(np.float32)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        obs = super()._get_obs()
        obs["achieved_goal"] = np.concatenate((obs["achieved_goal"], self.goal[3:6]))
        return obs

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id("target0")
        self.sim.model.site_pos[site_id] = self.goal[:3] - sites_offset[0]
        # Visualize obstacle
        site_id = self.sim.model.site_name2id("obstacle0")
        self.sim.model.site_pos[site_id] = self.goal[3:6] - sites_offset[0]
        self.sim.forward()
