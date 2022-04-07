from gym import utils
import numpy as np

from envs.parallel_jaw.flat_base import FlatPJBase
from envs.rotations import quatmultiply, axisangle2quat


class FlatPJCylinder(FlatPJBase, utils.EzPickle):

    def __init__(self, p_rot: float = 0.75, obj_size_range: float = 0):
        FlatPJBase.__init__(self, obj_name="cylinder", obj_size_range=obj_size_range)
        self.p_rot = p_rot
        utils.EzPickle.__init__(self, p_rot=p_rot, obj_size_range=obj_size_range)

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
        # Random rotation around z axis
        if self.np_random.rand() < self.p_rot:
            qy = axisangle2quat(0, 1, 0, np.pi / 2)
            qz = axisangle2quat(0, 0, 1, self.np_random.rand() * np.pi)
            q = quatmultiply(qz, qy)
        else:
            q = axisangle2quat(0, 0, 1, self.np_random.rand() * np.pi)
        object_qpos[3:7] = q
        self.sim.data.set_joint_qpos(self.obj_name + ":joint", object_qpos)
        self.sim.forward()
        return True
