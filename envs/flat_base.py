"""Flat desk fetch base class file."""
from typing import Dict, List, Optional
import logging

import numpy as np

import envs.rotations
import envs.robot_env
import envs.utils

logger = logging.getLogger(__name__)


class FlatBase(envs.robot_env.RobotEnv):
    """Superclass for all flat desk environments."""

    def __init__(self,
                 model_xml_path: str,
                 gripper_extra_height: float,
                 initial_qpos: dict,
                 n_actions: int,
                 object_name,
                 object_size_range,
                 initial_gripper: Optional[List] = None):
        """Initialize a new flat environment.

        Args:
            model_xml_path: Path to the Mujoco xml description
            gripper_extra_height: additional height above the table when positioning the gripper
            initial_qpos: a dictionary of joint names and values defining the initial configuration
            n_actions: Action state dimension
            object_name: Name of the manipulation object in Mujoco
            object_size_range: Range of object size modification. If 0, modification is disabled.
        """
        self.gripper_extra_height = gripper_extra_height
        self.object_range = np.array([0.1, 0.15])  # Admissible object range from the table center
        self.target_range = 0.15  # Admissible target range from the table center
        self.target_threshold = 0.05  # Range tolerance for task completion
        self.n_actions = n_actions
        self.object_name = object_name
        self.gripper_init_pos = None  # Overwritten during first _env_reset() call
        self.gripper_init_range = 0.15  # Admissable range from gripper_init_pos
        self.gripper_start_pos = None  # Current starting position of the gripper
        self.height_offset = 0.43
        self.initial_qpos = initial_qpos
        self.initial_gripper = initial_gripper
        self.object_size_range = object_size_range
        self.object_init_size = {}  # Save object sizes before modification
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
        return -(d > self.target_threshold).astype(np.float32)

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
        # Randomize start position of object
        object_pose = self._sample_object_pose()
        self.sim.data.set_joint_qpos(self.object_name + ":joint", object_pose)
        self.sim.forward()
        return True

    def _sample_object_pose(self) -> np.ndarray:
        object_pos = self.gripper_start_pos[:2]
        while np.linalg.norm(object_pos - self.gripper_start_pos[:2]) < 0.1:
            object_pos = self.sim.data.get_body_xpos("table0")[:2] + self.np_random.uniform(
                -self.object_range, self.object_range)
        object_pose = self.sim.data.get_joint_qpos(self.object_name + ":joint")
        assert object_pose.shape == (7,)
        object_pose[:2] = object_pos
        return object_pose

    def _sample_goal(self) -> np.ndarray:
        goal = self.sim.data.get_body_xpos("table0")[:3] + self.np_random.uniform(
            -self.target_range, self.target_range, size=3)
        while np.linalg.norm(
                self.sim.data.get_joint_qpos(self.object_name + ":joint")[:2] - goal[:2]) < 0.1:
            goal = self.sim.data.get_body_xpos("table0")[:3] + self.np_random.uniform(
                -self.target_range, self.target_range, size=3)
        goal[2] = self.height_offset
        if self.np_random.uniform() < 0.5:
            goal[2] += self.np_random.uniform(0, 0.45)
        return goal.copy()

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        d = envs.utils.goal_distance(achieved_goal, desired_goal)
        return (d < self.target_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos: np.ndarray):
        self._modify_object_size()
        for name, value in initial_qpos.items():
            if name not in self.sim.model.joint_names:
                logger.warning(f"Joint {name} present in initial_qpos, but not in Mujoco!")
                continue
            self.sim.data.set_joint_qpos(name, value)
        envs.utils.reset_mocap_welds(self.sim)
        self.sim.forward()
        # Save start positions on first run
        if self.gripper_init_pos is None:
            self.gripper_init_pos = self.sim.data.get_site_xpos("robot0:grip").copy()
        # Move end effector into position
        self._set_gripper_pose()
        # Change object pose
        self._set_object_pose()
        # Run sim
        for _ in range(10):
            if self.initial_gripper:
                self.sim.data.ctrl[:] = self.initial_gripper
            self._get_viewer(mode="human").render()
            self.sim.step()
        # Extract information for sampling goals
        self.gripper_start_pos = self.sim.data.get_site_xpos("robot0:grip").copy()

    def _set_gripper_pose(self):
        gripper_pos = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height
                               ]) + self.gripper_init_pos  # noqa: E124
        d_pos = self.np_random.uniform(-self.gripper_init_range, self.gripper_init_range, size=2)
        gripper_pos[:2] += d_pos  # Add random initial position change
        gripper_rot = np.array([1.0, 0.0, 1.0, 0.0])
        d_rot = self.np_random.uniform(-1, 1, size=4)
        # gripper_rot += (d_rot / np.linalg.norm(d_rot)) * 0.2  # Add random initial rotation change
        gripper_rot /= np.linalg.norm(gripper_rot)  # Renormalize for quaternion
        self.sim.data.set_mocap_pos("robot0:mocap", gripper_pos)
        self.sim.data.set_mocap_quat("robot0:mocap", gripper_rot)

    def _set_object_pose(self):
        object_pose = self.sim.data.get_joint_qpos(self.object_name + ":joint")
        object_pose[:2] = self.sim.data.get_body_xpos("table0")[:2]
        object_pose[2] = self.height_offset
        self.sim.data.set_joint_qpos(self.object_name + ":joint", object_pose)

    def _modify_object_size(self):
        if self.object_size_range > 0:
            # Reset previously modified object sizes first
            for object_name in self.object_init_size.keys():
                idx = self.sim.model.geom_name2id(object_name)
                self.sim.model.geom_size[idx] = self.object_init_size[object_name]
            if self.object_name == "mesh":
                return
            # Set new model size
            idx = self.sim.model.geom_name2id(self.object_name)
            # Save original size before modifying
            if self.object_name not in self.object_init_size.keys():
                self.object_init_size[self.object_name] = self.sim.model.geom_size[idx].copy()
            if self.object_name in ("cube", "cylinder", "sphere"):
                dsize = self.np_random.uniform(-self.object_size_range,
                                               self.object_size_range,
                                               size=3)
                self.sim.model.geom_size[idx] = self.object_init_size[self.object_name] + dsize
            else:
                raise RuntimeError(f"Object {self.object_name} not supported")
