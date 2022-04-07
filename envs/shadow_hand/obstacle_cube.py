from typing import Optional, Dict
from pathlib import Path

import numpy as np
from gym import utils

import envs
from envs.utils import goal_distance
from envs.shadow_hand.flat_base import FlatSHBase

MODEL_XML_PATH = str(Path("sh", "obstacle_pick_and_place.xml"))


class ObstacleSHCube(FlatSHBase, utils.EzPickle):

    def __init__(self, n_eigengrasps: Optional[int]):
        """Initialize a new flat environment.

        Args:
            obj_name: Name of the manipulation object in Mujoco
            n_eigengrasps: Number of eigengrasps to use
        """
        self.obstacle_threshold = 0.1
        super().__init__(model_xml_path=MODEL_XML_PATH,
                         obj_name="cube",
                         n_eigengrasps=n_eigengrasps)
        utils.EzPickle.__init__(self, n_eigengrasps=n_eigengrasps)

    def _env_setup(self, initial_qpos: np.ndarray):
        for name, value in initial_qpos.items():
            try:
                self.sim.data.set_joint_qpos(name, value)
            except ValueError:
                ...  # Some objects in initial_qpos are not in the obstacle XML
        envs.utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        if self.gripper_init_xpos is None:
            self.gripper_init_xpos = self.sim.data.get_site_xpos("robot0:grip").copy()
        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height
                                  ]) + self.gripper_init_xpos  # noqa: E124
        dpos = self.np_random.uniform(-self.gripper_init_range, self.gripper_init_range, size=2)
        gripper_target[:2] += dpos  # Add random initial position change

        gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        drot = self.np_random.uniform(-1, 1, size=4)
        gripper_rotation += (drot /
                             np.linalg.norm(drot)) * 0.2  # Add random initial orientation change
        gripper_rotation /= np.linalg.norm(gripper_rotation)  # Renormalize for quaternion
        self.sim.data.set_mocap_pos("robot0:mocap", gripper_target)
        self.sim.data.set_mocap_quat("robot0:mocap", gripper_rotation)

        # Set a new object position
        object_xpos = gripper_target[:2]
        while np.linalg.norm(object_xpos - gripper_target[:2]) < 0.1:
            object_xpos = self.sim.data.get_body_xpos("table0")[:2] + self.np_random.uniform(
                -self.obj_range, self.obj_range, size=2)
        object_qpos = self.sim.data.get_joint_qpos(self.obj_name + ":joint")
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        object_qpos[2] = self.height
        object_qpos[3:7] = (self.np_random.rand(4) - 0.5) * 2  # Random initial orientation
        object_qpos[3:7] /= np.linalg.norm(object_qpos[3:7])
        self.sim.data.set_joint_qpos(self.obj_name + ":joint", object_qpos)
        t = 0
        while np.linalg.norm(self.sim.data.get_site_xvelp(self.obj_name)) > 1e-2 or t < 10:
            self.sim.step()
            t += 1
        obj_pos = self.sim.data.get_site_xpos(self.obj_name)[:2]
        if any(abs(obj_pos - self.sim.data.get_body_xpos("table0")[:2]) > self.obj_range):
            return self._env_setup(initial_qpos)  # Retry if object is out of bounds
        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos("robot0:grip").copy()
        self.height_offset = self.sim.data.get_site_xpos("target0")[2]

    def _reset_sim(self):
        self._env_setup(self.initial_qpos)
        return True

    def compute_reward(self, achieved_goal: np.ndarray, goal: np.ndarray, _) -> np.ndarray:
        """Compute the agent reward for the achieved goal.

        Args:
            achieved_goal: Achieved goal.
            goal: Desired goal.
        """
        # Compute distance between goal and the achieved goal.
        if goal.ndim == 2:
            goal_d = goal_distance(achieved_goal[:, :3], goal[:, :3])
            obstacle_d = goal_distance(achieved_goal[:, 3:6], goal[:, 3:6])
        else:
            goal_d = goal_distance(achieved_goal[:3], goal[:3])
            obstacle_d = goal_distance(achieved_goal[3:6], goal[3:6])
        goal_reward = -(goal_d > self.distance_threshold).astype(np.float32)
        obstacle_reward = -(obstacle_d > self.obstacle_threshold).astype(np.float32)
        return goal_reward + obstacle_reward

    def _sample_goal(self) -> np.ndarray:
        goal = self.sim.data.get_body_xpos("table0")[:3] + self.np_random.uniform(
            -self.target_range, self.target_range, size=3)
        goal += self.target_offset
        goal[2] = self.height_offset
        if self.np_random.uniform() < 0.5:
            goal[2] += self.np_random.uniform(0, 0.45)
        # Sample obstacle outside of hand and obstacle
        obstacle = self.sim.data.get_body_xpos("table0")[:3] + self.np_random.uniform(
            -self.target_range, self.target_range, size=3)
        obstacle[2] = self.height
        obj_xpos = self.sim.data.get_site_xpos(self.obj_name)
        while goal_distance(obstacle, goal) < 0.15 or goal_distance(
                self.initial_gripper_xpos, obstacle) < 0.15 or goal_distance(obstacle,
                                                                             obj_xpos) < 0.15:
            obstacle = self.sim.data.get_body_xpos("table0")[:3] + self.np_random.uniform(
                -self.target_range, self.target_range, size=3)
            obstacle[2] = self.height
        return np.concatenate((goal, obstacle))

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        d = goal_distance(achieved_goal[:3], desired_goal[:3])
        return (d < self.distance_threshold).astype(np.float32)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        # positions
        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip") * dt
        robot_qpos, robot_qvel = envs.utils.robot_get_obs(self.sim)
        object_pos = self.sim.data.get_site_xpos(self.obj_name)
        # rotations
        object_rot = envs.rotations.mat2euler(self.sim.data.get_site_xmat(self.obj_name))
        # velocities
        object_velp = self.sim.data.get_site_xvelp(self.obj_name) * dt
        object_velr = self.sim.data.get_site_xvelr(self.obj_name) * dt
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
            "achieved_goal": np.concatenate((achieved_goal, self.goal[3:6])),
            "desired_goal": self.goal.copy(),
        }

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id("target0")
        self.sim.model.site_pos[site_id] = self.goal[:3] - sites_offset[0]
        # Visualize obstacle
        site_id = self.sim.model.site_name2id("obstacle0")
        self.sim.model.site_pos[site_id] = self.goal[3:6] - sites_offset[0]
        self.sim.forward()
