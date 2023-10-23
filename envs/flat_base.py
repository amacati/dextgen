"""Grasp base environment module.

The overall module structure is based on OpenAI's fetch environments, but the module itself has
largely been rewritten for our purpose. This also applies to all environments in ``barrett_hand``,
``parallel_jaw`` and ``shadow_hand``.

See https://github.com/Farama-Foundation/Gym-Robotics/tree/main/gym_robotics/envs.
"""
from typing import Dict, List, Optional
import logging
import copy

import numpy as np
import mujoco_py

import envs.rotations
import envs.robot_env
import envs.utils

logger = logging.getLogger(__name__)


class FlatBase(envs.robot_env.RobotEnv):
    """Base class for all grasp environments."""

    n_substeps = 20

    def __init__(self,
                 model_xml_path: str,
                 gripper_extra_height: float,
                 initial_qpos: dict,
                 n_actions: int,
                 object_name: str,
                 object_size_multiplier: float = 1.,
                 object_size_range: float = 0.,
                 initial_gripper: Optional[List] = None):
        """Initialize a grasp environment.

        Args:
            model_xml_path: Path to the Mujoco xml description
            gripper_extra_height: additional height above the table when positioning the gripper
            initial_qpos: a dictionary of joint names and values defining the initial configuration
            n_actions: Action state dimension
            object_name: Name of the manipulation object in Mujoco
            object_size_multiplier: Optional multiplier to change object sizes by a fixed amount.
            object_size_range: Optional range to randomly enlarge/shrink object sizes.
            initial_gripper: Default initial gripper joint positions.
        """
        self.gripper_extra_height = gripper_extra_height
        self.gripper_init_pos = None
        self.object_range = np.array([0.1, 0.15])  # Admissible object range from the table center
        self.target_range = 0.15  # Admissible target range from the table center
        self.target_threshold = 0.05  # Range tolerance for task completion
        self.n_actions = n_actions
        self.object_name = object_name
        self.gripper_init_range = np.array([0.05, 0.1])  # Admissable range from gripper_init_pos
        self.gripper_start_pos = None  # Current starting position of the gripper
        self.height_offset = 0.43
        self.goal_max_height = 0.3
        self.initial_qpos = initial_qpos
        self.initial_gripper = initial_gripper
        self.early_stop_ok = True  # Flag to prevent an early stop
        self._reset_sim_state = None
        self._reset_sim_goal = None
        self.p_high_goal = 0  # Probability of a high goal
        assert object_size_multiplier > 0
        self.object_size_multiplier = object_size_multiplier
        assert object_size_range >= 0
        self.object_size_range = object_size_range
        super().__init__(
            model_path=model_xml_path,
            n_substeps=self.n_substeps,
            n_actions=n_actions,
            initial_qpos=initial_qpos,
        )
        self.object_init_size = {}  # Save object sizes before modification

    def compute_reward(self, achieved_goal: np.ndarray, goal: np.ndarray, _) -> np.ndarray:
        """Compute the agent reward for the achieved goal.

        Args:
            achieved_goal: Achieved goal.
            goal: Desired goal.
        """
        # Compute distance between goal and the achieved goal.
        d = envs.utils.goal_distance(achieved_goal[..., :3], goal[..., :3])
        return -(d > self.target_threshold).astype(np.float32)

    def _set_action(self, action: np.ndarray):
        """`_set_action` is robot specific, implement in child classes."""
        raise NotImplementedError

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """`_get_obs` is robot specific, implement in child classes."""
        raise NotImplementedError()

    def _get_contact_info(self) -> Dict:
        contact_info = []
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            geom1 = self.sim.model.geom_id2name(contact.geom1)
            geom2 = self.sim.model.geom_id2name(contact.geom2)
            if geom1 is None or geom2 is None:
                continue
            if self.object_name in (geom1, geom2) and ("robot0" in geom1 or "robot0" in geom2):
                # Always have the gripper link as the first geometry and the object as the second
                geom1 = geom2 if geom1 == self.object_name else geom1
                geom2 = self.object_name
                contact_force = np.zeros(6, dtype=np.float64)
                # Contact force is 3 forces + 3 torques
                mujoco_py.functions.mj_contactForce(self.sim.model, self.sim.data, i, contact_force)
                # MuJoCo stores contact matrices transposed.
                # See https://mujoco.readthedocs.io/en/latest/programming.html#contacts
                frame = contact.frame.reshape(-3, 3).T
                info = {
                    "geom1": geom1,
                    "geom2": geom2,
                    "contact_force": contact_force,
                    "pos": contact.pos.copy(),
                    "frame": frame.copy()
                }
                contact_info.append(info)
        return contact_info

    def _get_gripper_info(self) -> Dict:
        pos = self.sim.data.get_site_xpos("robot0:grip").copy()
        orient = self.sim.data.get_site_xmat("robot0:grip").copy()
        robot_qpos, _ = envs.utils.robot_get_obs(self.sim)
        if self.gripper_type == "ParallelJaw":
            state = robot_qpos[-2:].copy()
        elif self.gripper_type == "BarrettHand":
            state = robot_qpos[-8:].copy()
        elif self.gripper_type == "ShadowHand":
            state = robot_qpos[-24:]
        elif self.gripper_type == "SeaClear":
            state = robot_qpos[-2].copy()
        else:
            raise RuntimeError("Gripper type not supported")
        return {"pos": pos, "orient": orient, "state": state, "type": self.gripper_type}

    def _get_object_info(self) -> Dict:
        object_pos = self.sim.data.get_site_xpos(self.object_name).copy()
        object_orient = self.sim.data.get_site_xmat(self.object_name).copy()
        object_size = self.sim.model.geom_size[self.sim.model.geom_name2id(self.object_name)]
        return {
            "pos": object_pos,
            "orient": object_orient,
            "name": self.object_name,
            "size": object_size
        }

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
        site_id = self.sim.model.site_name2id("target0")
        self.sim.model.site_pos[site_id] = self.goal[:3]
        self.sim.forward()

    def save_reset(self):
        """Save the current environment state to load it with :meth:`~.save_reset`.

        Note: Has to be called at least once before loading a reset point.
        Warning: Experimental. Only introduced for optimization.
        """
        self._reset_sim_state = copy.deepcopy(self.sim.get_state())
        self._reset_sim_goal = self.goal.copy()

    def load_reset(self) -> Dict[str, np.ndarray]:
        """Load a saved environment state.

        Warning: :meth:`~.reset` still has to be called before loading the state!
        Warning: Experimental. Only introduced for optimization.
        """
        assert self._reset_sim_state is not None and self._reset_sim_goal is not None
        self.reset()  # Somehow not sufficient to reset the steps counter
        self.sim.set_state(self._reset_sim_state)
        self.goal = self._reset_sim_goal
        self.sim.forward()
        return self._get_obs()

    def _reset_sim(self) -> bool:
        self.sim.set_state(self.initial_state)
        self._env_setup(self.initial_qpos)  # Rerun env setup to get new start poses for the robot
        # Randomize start position of object
        object_pose = self._sample_object_pose()
        self.sim.data.set_joint_qpos(self.object_name + ":joint", object_pose)
        self.sim.forward()
        return True

    def _sample_object_pose(self) -> np.ndarray:
        object_pos = self.sim.data.get_body_xpos("table0")[:2] + self.np_random.uniform(
            -self.object_range, self.object_range)
        while np.linalg.norm(object_pos - self.gripper_start_pos[:2]) < 0.1:
            object_pos = self.sim.data.get_body_xpos("table0")[:2] + self.np_random.uniform(
                -self.object_range, self.object_range)
        object_pose = self.sim.data.get_joint_qpos(self.object_name + ":joint")
        assert object_pose.shape == (7,)
        object_pose[:2] = object_pos
        return object_pose

    def _sample_goal(self) -> np.ndarray:
        table_pos = self.sim.data.get_body_xpos("table0")[:3]
        object_pos = self.sim.data.get_site_xpos(self.object_name)[:3]
        goal = table_pos.copy()
        goal[:2] += self.np_random.uniform(-self.target_range, self.target_range, size=2)
        while np.linalg.norm(object_pos[:2] - goal[:2]) < 0.1:
            goal = table_pos.copy()
            goal[:2] += self.np_random.uniform(-self.target_range, self.target_range, size=2)
        goal[2] = self.height_offset
        # Random goal height
        if self.np_random.uniform() < self.p_high_goal:
            goal[2] += self.np_random.uniform(self.goal_max_height / 2, self.goal_max_height)
        return goal.copy()

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        d = envs.utils.goal_distance(achieved_goal, desired_goal)
        return (d < self.target_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos: np.ndarray):
        self._modify_object_size()
        for name, value in initial_qpos.items():
            # Envs have joint start values for multiple grasp objects. Objects are not present in
            # all MuJoCo envs to speed up simulation for single grasp object environments. Therefore
            # non-existent joints are expected
            if name not in self.sim.model.joint_names:
                continue
            self.sim.data.set_joint_qpos(name, value)
        envs.utils.reset_mocap_welds(self.sim)
        self.sim.forward()
        # Save start positions on first run
        if self.gripper_init_pos is None:
            self.gripper_init_pos = self.sim.data.get_body_xpos("table0")[:3].copy()
            self.gripper_init_pos[2] = 0.4 + self.gripper_extra_height  # Table height + offset
        # Move end effector into position
        self._set_gripper_pose()
        # Change object pose
        self._set_object_pose()
        # Run sim
        for _ in range(10):
            if self.initial_gripper:
                self.sim.data.ctrl[:] = self.initial_gripper
            self.sim.step()
        # Extract information for sampling goals
        self.gripper_start_pos = self.sim.data.get_site_xpos("robot0:grip").copy()

    def _set_gripper_pose(self):
        gripper_pos = self.gripper_init_pos.copy()  # noqa: E124
        d_pos = self.np_random.uniform(-self.gripper_init_range, self.gripper_init_range)
        gripper_pos[:2] += d_pos  # Add random initial position change
        gripper_rot = np.array([1.0, 0.0, 1.0, 0.0])
        # d_rot = self.np_random.uniform(-1, 1, size=4)
        # gripper_rot += (d_rot / np.linalg.norm(d_rot)) * 0.2  # Add random initial rotation change
        # gripper_rot /= np.linalg.norm(gripper_rot)  # Renormalize for quaternion
        self.sim.data.set_mocap_pos("robot0:mocap", gripper_pos)
        self.sim.data.set_mocap_quat("robot0:mocap", gripper_rot)

    def _set_object_pose(self):
        object_pose = self.sim.data.get_joint_qpos(self.object_name + ":joint")
        object_pose[:2] = self.sim.data.get_body_xpos("table0")[:2]
        object_pose[2] = self.height_offset
        self.sim.data.set_joint_qpos(self.object_name + ":joint", object_pose)

    def _modify_object_size(self):
        if self.object_size_range > 0 or self.object_size_multiplier != 1:
            supported_objects = ("cube", "cylinder", "sphere")
            if self.object_name not in supported_objects:
                raise RuntimeError(f"Object {self.object_name} does not support size modification!")
            # Reset previously modified object sizes first
            for object_name in self.object_init_size.keys():
                idx = self.sim.model.geom_name2id(object_name)
                self.sim.model.geom_size[idx] = self.object_init_size[object_name]
            # if self.object_name == "mesh":
            #    return
            # Set new model size
            idx = self.sim.model.geom_name2id(self.object_name)
            # Save original size before modifying
            if self.object_name not in self.object_init_size.keys():
                if self.object_size_multiplier != 1:
                    # Modify initial save to include the multiplier already. Necessary when
                    # combining object size multiplication and range at the same time.
                    geom_size = self.sim.model.geom_size[idx] * self.object_size_multiplier
                    self.object_init_size[self.object_name] = geom_size
                else:
                    self.object_init_size[self.object_name] = self.sim.model.geom_size[idx].copy()
            geom_size = self.object_init_size[self.object_name]
            if self.object_size_range > 0:
                geom_range = self.object_size_range
                geom_size += self.np_random.uniform(-geom_range, geom_range, size=2)
            self.sim.model.geom_size[idx] = geom_size
