"""Flat desk fetch base class file."""
from typing import Union, Dict
from pathlib import Path

import numpy as np

import envs.rotations
import envs.robot_env
import envs.utils


class FlatBase(envs.robot_env.RobotEnv):
    """Superclass for all flat desk environments."""

    def __init__(self, model_xml_path: str, gripper_extra_height: float,
                 target_offset: Union[float, np.ndarray], initial_qpos: dict, n_actions: int,
                 obj_name, obj_size_range):
        """Initialize a new flat environment.

        Args:
            model_xml_path: Path to the Mujoco xml description
            gripper_extra_height: additional height above the table when positioning the gripper
            target_offset: offset of the target
            initial_qpos: a dictionary of joint names and values defining the initial configuration
            n_actions: Action state dimension
            obj_name: Name of the manipulation object in Mujoco
            obj_size_range: Range of object size modification. If 0, modification is disabled.
        """
        self.gripper_extra_height = gripper_extra_height
        self.target_offset = target_offset
        self.obj_range = np.array([0.1, 0.15])
        self.target_range = 0.15
        self.distance_threshold = 0.05
        self.n_actions = n_actions
        self.obj_name = obj_name
        self.gripper_init_range = 0.15
        self.height = 0.43
        self.initial_qpos = initial_qpos
        self.gripper_init_xpos = None  # Save once during init
        self.obj_size_range = obj_size_range
        self.obj_init_size = {}  # Save object sizes before modification
        super().__init__(
            model_path=model_xml_path,
            n_substeps=20,
            n_actions=n_actions,
            initial_qpos=initial_qpos,
        )

    def compute_reward(self, achieved_goal: np.ndarray, goal: np.ndarray, _) -> np.ndarray:
        """Compute the agent reward for the achieved goal.

        Args:
            achieved_goal: Achieved goal.
            goal: Desired goal.
        """
        # Compute distance between goal and the achieved goal.
        d = envs.utils.goal_distance(achieved_goal, goal)
        return -(d > self.distance_threshold).astype(np.float32)

    def _set_action(self, action: np.ndarray):
        """`_set_action` is robot specific, implement in child classes."""
        raise NotImplementedError

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """`_get_obs` is robot specific, implement in child classes."""
        raise NotImplementedError()

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id("robot0:gripper_link")
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.0
        self.viewer.cam.elevation = -14.0

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id("target0")
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self) -> bool:
        self._env_setup(self.initial_qpos)  # Rerun env setup to get new start poses for the robot
        # Randomize start position of object.
        object_xpos = self.initial_gripper_xpos[:2]
        while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
            object_xpos = self.sim.data.get_body_xpos("table0")[:2] + self.np_random.uniform(
                -self.obj_range, self.obj_range, size=2)
        object_qpos = self.sim.data.get_joint_qpos(self.obj_name + ":joint")
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        self.sim.data.set_joint_qpos(self.obj_name + ":joint", object_qpos)
        self.sim.forward()
        return True

    def _sample_goal(self) -> np.ndarray:
        goal = self.sim.data.get_body_xpos("table0")[:3] + self.np_random.uniform(
            -self.target_range, self.target_range, size=3)
        goal += self.target_offset
        goal[2] = self.height_offset
        if self.np_random.uniform() < 0.5:
            goal[2] += self.np_random.uniform(0, 0.45)
        return goal.copy()

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        d = envs.utils.goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos: np.ndarray):
        self._modify_object_size()
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
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
        object_qpos = self.sim.data.get_joint_qpos(self.obj_name + ":joint")
        object_qpos[:2] = self.sim.data.get_body_xpos("table0")[:2]
        object_qpos[2] = self.height
        self.sim.data.set_joint_qpos(self.obj_name + ":joint", object_qpos)
        for _ in range(10):
            self.sim.step()
        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos("robot0:grip").copy()
        self.height_offset = self.sim.data.get_site_xpos("target0")[2]

    def _modify_object_size(self):
        if self.obj_size_range > 0:
            # Reset object sizes first
            for obj_name in self.obj_init_size.keys():
                idx = self.sim.model.geom_name2id(obj_name)
                self.sim.model.geom_size[idx] = self.obj_init_size[obj_name]
            if self.obj_name == "mesh":
                return
            # Set new model size
            idx = self.sim.model.geom_name2id(self.obj_name)
            if self.obj_name not in self.obj_init_size.keys():  # Add original size before modifying
                self.obj_init_size[self.obj_name] = self.sim.model.geom_size[idx].copy()
            if self.obj_name in ("cube", "cylinder", "sphere"):
                dsize = self.np_random.uniform(-self.obj_size_range, self.obj_size_range, size=3)
                self.sim.model.geom_size[idx] = self.obj_init_size[self.obj_name] + dsize
            else:
                raise RuntimeError(f"Object {self.obj_name} not supported")
