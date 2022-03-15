"""ObstacleReach environment class file."""
import numpy as np
from pathlib import Path
from typing import Dict
from gym import utils
import envs.utils
from envs.utils import goal_distance
from envs.parallel_jaw.fetch import FetchEnv

MODEL_XML_PATH = str(Path("fetch", "obstaclereach.xml"))


class ObstacleReach(FetchEnv, utils.EzPickle):
    """Environment for reaching a goal with a forbidden region that should not be crossed."""

    def __init__(self, reward_type: str = "sparse", obstacle_threshold: float = 0.15):
        """Initialize the Mujoco sim."""
        initial_qpos = {
            "robot0:slide0": 0.4049,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
        }
        self.obstacle_threshold = obstacle_threshold
        self.height_offset = 0.4
        self.c_low = (1.05, 0.4, 0.4)
        self.c_high = (1.55, 1.1, 0.4)
        FetchEnv.__init__(
            self,
            MODEL_XML_PATH,
            has_object=False,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
        )
        utils.EzPickle.__init__(self,
                                reward_type=reward_type,
                                obstacle_threshold=obstacle_threshold)

    def compute_reward(self, achieved_goal: np.ndarray, goal: np.ndarray, _) -> np.ndarray:
        """Compute the agent reward for the achieved goal.

        Args:
            achieved_goal: Achieved goal.
            goal: Desired goal.
        """
        if goal.ndim == 2:
            dg = goal_distance(achieved_goal[:, :3], goal[:, :3])
            dff = goal_distance(achieved_goal[:, 3:], goal[:, 3:])
        else:
            dg = goal_distance(achieved_goal[:3], goal[:3])
            dff = goal_distance(achieved_goal[3:], goal[3:])
        if self.reward_type == "sparse":
            reward_goal = -(dg > self.distance_threshold).astype(np.float32)
            reward_obstacle = -(dff > self.obstacle_threshold).astype(np.float32)
            return reward_goal + reward_obstacle
        return -dg + .2 * dff

    def _get_obs(self) -> Dict[str, np.ndarray]:
        # positions
        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip") * dt
        robot_qpos, robot_qvel = envs.utils.robot_get_obs(self.sim)
        object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric
        # Redefine the achieved obstacle goal if the gripper entered the original one
        obstacle_region = self.goal[3:].copy()
        while goal_distance(obstacle_region, grip_pos) < self.obstacle_threshold:
            obstacle_region = self.np_random.uniform(-self.target_range, self.target_range, size=3)
        achieved_goal = np.concatenate([grip_pos.copy(), obstacle_region])
        obs = np.concatenate([
            grip_pos,
            object_pos.ravel(),
            object_rel_pos.ravel(),
            gripper_state,
            object_rot.ravel(),
            object_velp.ravel(),
            object_velr.ravel(),
            grip_velp,
            gripper_vel,
        ])

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }

    def _sample_goal(self) -> np.ndarray:
        goal = self.np_random.uniform(self.c_low, self.c_high)
        while goal_distance(self.initial_gripper_xpos[:3], goal) < 3 * self.distance_threshold:
            goal = self.np_random.uniform(self.c_low, self.c_high)
        obstacle = self.np_random.uniform(self.c_low, self.c_high)
        while goal_distance(
                self.initial_gripper_xpos[:3], obstacle) < self.obstacle_threshold or goal_distance(
                    goal, obstacle) < self.obstacle_threshold:
            obstacle = self.np_random.uniform(self.c_low, self.c_high)
        return np.concatenate([goal, obstacle]).copy()

    def _render_callback(self):
        # Visualize target and obstacle region
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id("target0")
        self.sim.model.site_pos[site_id] = self.goal[0:3] - sites_offset[0]
        site_id = self.sim.model.site_name2id("obstacle")
        self.sim.model.site_pos[site_id] = self.goal[3:6] - sites_offset[0]
        self.sim.forward()
