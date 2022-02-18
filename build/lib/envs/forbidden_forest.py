import numpy as np
from pathlib import Path
import gym
from gym import utils
import rotations
from fetch import FetchEnv, goal_distance

MODEL_XML_PATH = Path("fetch", "reach.xml")


class ForbiddenForest(FetchEnv, utils.EzPickle):

    def __init__(self, reward_type="sparse"):
        initial_qpos = {
            "robot0:slide0": 0.4049,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
        }
        FetchEnv.__init__(
            self,
            MODEL_XML_PATH,
            has_object=False,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
        )
        utils.EzPickle.__init__(self, reward_type=reward_type)

    def compute_reward(self, achieved_goal, goal, info):
        dg = goal_distance(achieved_goal[0:3], goal[0:3])
        dff = goal_distance(achieved_goal[3:6], goal[3:6])
        if self.reward_type == "sparse":
            reward_goal = -(dg > self.distance_threshold).astype(np.float32)
            reward_forbidden_forest = -(dff > self.ff_distance_threshold).astype(np.float32)
            return reward_goal + reward_forbidden_forest
        return -dg + .2 * dff

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip") * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric
        achieved_goal = np.concatenate([grip_pos.copy(), self.forbidden.copy()])
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

    def _sample_goal(self):
        goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
            -self.target_range, self.target_range, size=3)
        forbidden = self.initial_gripper_xpos[:3] + self.np_random.uniform(
            -self.target_range, self.target_range, size=3)
        return np.concatenate([goal, forbidden]).copy()

    def _render_callback(self):
        # Visualize target and forbidden region
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id("target0")
        self.sim.model.site_pos[site_id] = self.goal[0:3] - sites_offset[0]
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id("target1")
        self.sim.model.site_pos[site_id] = self.goal[3:6] - sites_offset[0]
        self.sim.forward()
