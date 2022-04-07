from typing import Dict
from pathlib import Path

import numpy as np

import envs
from envs.flat_base import FlatBase
from envs.rotations import axisangle2quat
from envs.utils import goal_distance

MODEL_XML_PATH = str(Path("seaclear", "seaclear.xml"))


class SeaClear(FlatBase):

    def __init__(self):
        """Initialize a new flat environment.

        Args:
            obj_name: Name of the manipulation object in Mujoco
        """
        self.mocap_offset = np.array([0., 0., 0.2])
        self.p_target_in_air = 0.5
        self._obs_dist = 0.15
        self._obs_threshold = 0.1
        super().__init__(model_xml_path=MODEL_XML_PATH,
                         gripper_extra_height=0.2,
                         target_offset=0.,
                         initial_qpos={},
                         obj_name="can",
                         n_actions=5,
                         obj_size_range=0)
        self.target_range = 0.3

    def _set_action(self, action):
        assert action.shape == (self.n_actions,)
        action = (action.copy())  # ensure that we don't change the action outside of this scope
        pos_ctrl, rot_ctrl, gripper_ctrl = action[:3], action[3], action[4]
        pos_ctrl[2] = -0.1
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
        gripper_state = robot_qpos[-2:]
        gripper_vel = (robot_qvel[-2:] * dt)  # change to a scalar if the gripper is made symmetric

        achieved_goal = np.squeeze(object_pos.copy())
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
            "achieved_goal": np.concatenate((achieved_goal, self.goal[3:6])),
            "desired_goal": self.goal.copy(),
        }

    def _env_setup(self, initial_qpos: np.ndarray):
        # for name, value in initial_qpos.items():
        #     self.sim.data.set_joint_qpos(name, value)
        # envs.utils.reset_mocap_welds(self.sim)
        self.sim.forward()
        if self.gripper_init_xpos is None:
            self.gripper_init_xpos = self.sim.data.get_site_xpos("robot0:grip").copy()
        self.initial_gripper_xpos = self.sim.data.get_site_xpos("robot0:grip").copy()

    def compute_reward(self, achieved_goal: np.ndarray, goal: np.ndarray, _) -> np.ndarray:
        """Compute the agent reward for the achieved goal.

        Args:
            achieved_goal: Achieved goal.
            goal: Desired goal.
        """
        if goal.ndim == 2:
            goal_d = goal_distance(achieved_goal[:, :3], goal[:, :3])
            obstacle_d = goal_distance(achieved_goal[:, 3:6], goal[:, 3:6])
        else:
            goal_d = goal_distance(achieved_goal[:3], goal[:3])
            obstacle_d = goal_distance(achieved_goal[3:6], goal[3:6])
        goal_reward = -(goal_d > self.distance_threshold).astype(np.float32)
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

    def _reset_sim(self) -> bool:
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        obj_pose = np.zeros(7)
        obj_pose[:2] = self.np_random.uniform(-self.target_range, self.target_range, size=2)
        obj_pose[3] = 1  # Unit quaternion rotation
        self.sim.data.set_joint_qpos(self.obj_name + ":joint", obj_pose)
        self.sim.forward()
        return True

    def _sample_goal(self) -> np.ndarray:
        goal = self.np_random.uniform(-self.target_range, self.target_range, size=3)
        goal[2] = 0
        if self.np_random.rand() < self.p_target_in_air:
            goal[2] += self.np_random.uniform(0, 0.45)
        obstacle = self.np_random.uniform(-self.target_offset, self.target_range, size=3)
        obstacle[2] = 0
        obj_xpos = self.sim.data.get_site_xpos(self.obj_name)
        d_obj = goal_distance(obstacle[:2], obj_xpos[:2])
        d_goal = goal_distance(obstacle[:2], goal[:2])
        d_grip = goal_distance(obstacle[:2], self.initial_gripper_xpos[:2])
        while any([d < self._obs_dist for d in (d_obj, d_goal, d_grip)]):
            obstacle = self.np_random.uniform(-self.target_range, self.target_range, size=3)
            obstacle[2] = 0
            d_obj = goal_distance(obstacle[:2], obj_xpos[:2])
            d_goal = goal_distance(obstacle[:2], goal[:2])
            d_grip = goal_distance(obstacle[:2], self.initial_gripper_xpos[:2])
        return np.concatenate((goal, obstacle))

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        d = envs.utils.goal_distance(achieved_goal[:3], desired_goal[:3])
        return (d < self.distance_threshold).astype(np.float32)
