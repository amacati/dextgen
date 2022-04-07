from typing import Optional
from pathlib import Path

import numpy as np

import envs
from envs.shadow_hand.flat_base import FlatSHBase

MODEL_XML_PATH = str(Path("sh", "uneven_pick_and_place.xml"))


class UnevenSHBase(FlatSHBase):

    def __init__(self, obj_name: str, n_eigengrasps: Optional[int]):
        """Initialize a new flat environment.

        Args:
            obj_name: Name of the manipulation object in Mujoco
            n_eigengrasps: Number of eigengrasps to use
        """
        super().__init__(model_xml_path=MODEL_XML_PATH,
                         obj_name=obj_name,
                         n_eigengrasps=n_eigengrasps)

    def _env_setup(self, initial_qpos: np.ndarray):
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
