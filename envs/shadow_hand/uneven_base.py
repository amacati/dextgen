from typing import Optional
from pathlib import Path

import numpy as np

import envs
from envs.shadow_hand.flat_base import FlatSHBase

MODEL_XML_PATH = str(Path("sh", "uneven_pick_and_place.xml"))


class UnevenSHBase(FlatSHBase):

    def __init__(self, object_name: str, n_eigengrasps: Optional[int]):
        """Initialize a new flat environment.

        Args:
            object_name: Name of the manipulation object in Mujoco
            n_eigengrasps: Number of eigengrasps to use
        """
        super().__init__(model_xml_path=MODEL_XML_PATH,
                         object_name=object_name,
                         n_eigengrasps=n_eigengrasps)

    def _env_setup(self, initial_qpos: np.ndarray):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        envs.utils.reset_mocap_welds(self.sim)
        self.sim.forward()
        # Save start positions on first run
        if self.gripper_init_pos is None:
            self.gripper_init_pos = self.sim.data.get_site_xpos("robot0:grip").copy()
        if self.height_offset is None:
            self.height_offset = self.sim.data.get_site_xpos("target0")[2]
        # Move end effector into position
        self._set_gripper_pose()
        # Change object pose
        self._set_object_pose()
        # Run until the object has settled down
        t = 0
        while np.linalg.norm(self.sim.data.get_site_xvelp(self.object_name)) > 1e-2 or t < 10:
            self.sim.step()
            t += 1
        object_pos = self.sim.data.get_site_xpos(self.object_name)[:2]
        if any(abs(object_pos - self.sim.data.get_body_xpos("table0")[:2]) > self.object_range):
            return self._env_setup(initial_qpos)  # Retry if object is out of bounds
        # Extract information for sampling goals.
        self.gripper_start_pos = self.sim.data.get_site_xpos("robot0:grip").copy()

    def _set_object_pose(self):
        object_pos = self.sim.data.get_body_xpos("table0")[:2]
        while np.linalg.norm(object_pos - self.sim.data.get_mocap_pos("robot0:mocap")[:2]) < 0.1:
            object_pos = self.sim.data.get_body_xpos("table0")[:2] + self.np_random.uniform(
                -self.object_range, self.object_range)
        object_pose = self.sim.data.get_joint_qpos(self.object_name + ":joint")
        assert object_pose.shape == (7,)
        object_pose[:2] = object_pos
        object_pose[2] = self.height_offset
        object_pose[3:7] = (self.np_random.rand(4) - 0.5) * 2  # Random initial orientation
        object_pose[3:7] /= np.linalg.norm(object_pose[3:7])
        self.sim.data.set_joint_qpos(self.object_name + ":joint", object_pose)

    def _reset_sim(self):
        self._env_setup(self.initial_qpos)
        return True
