from gym import utils
import numpy as np

from envs.shadow_hand.flat_base import FlatSHBase
from envs.rotations import axisangle2quat, quatmultiply
from envs.utils import reset_mocap_welds


class FlatSHMesh(FlatSHBase, utils.EzPickle):

    def __init__(self, n_eigengrasps: int, obj_size_range: float = 0):
        FlatSHBase.__init__(self,
                            obj_name="mesh",
                            n_eigengrasps=n_eigengrasps,
                            obj_size_range=obj_size_range)
        utils.EzPickle.__init__(self, n_eigengrasps=n_eigengrasps, obj_size_range=obj_size_range)

    def _env_setup(self, initial_qpos: np.ndarray):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        reset_mocap_welds(self.sim)
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
        # Rotate object
        if self.np_random.rand() < 0.5:
            qy = axisangle2quat(0, 1, 0, np.pi / 2)
            qz = axisangle2quat(0, 0, 1, self.np_random.rand() * np.pi)
            q = quatmultiply(qz, qy)
        else:
            q = axisangle2quat(0, 0, 1, self.np_random.rand() * np.pi)
        object_qpos[3:7] = q
        self.sim.data.set_joint_qpos(self.obj_name + ":joint", object_qpos)
        for _ in range(10):
            self.sim.step()
        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos("robot0:grip").copy()
        self.height_offset = self.sim.data.get_site_xpos("target0")[2]

    def _reset_sim(self) -> bool:
        self._env_setup(self.initial_qpos)  # Rerun env setup to get new start poses for the robot
        # Randomize start position of object
        object_xpos = self.sim.data.get_joint_qpos(self.obj_name + ":joint")[:2]
        while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
            object_xpos = self.sim.data.get_body_xpos("table0")[:2] + self.np_random.uniform(
                -self.obj_range, self.obj_range, size=2)
        # Random rotation already in `_env_setup`
        object_qpos = self.sim.data.get_joint_qpos(self.obj_name + ":joint")
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        self.sim.data.set_joint_qpos(self.obj_name + ":joint", object_qpos)
        self.sim.forward()
        return True
