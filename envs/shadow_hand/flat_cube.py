from gym import utils
import numpy as np

from envs.shadow_hand.flat_base import FlatSHBase


class FlatSHCube(FlatSHBase, utils.EzPickle):

    def __init__(self, n_eigengrasps: int, obj_size_range: float = 0):
        FlatSHBase.__init__(self,
                            obj_name="cube",
                            n_eigengrasps=n_eigengrasps,
                            obj_size_range=obj_size_range)
        utils.EzPickle.__init__(self, n_eigengrasps=n_eigengrasps, obj_size_range=obj_size_range)

    def _reset_sim(self) -> bool:
        self._env_setup(self.initial_qpos)  # Rerun env setup to get new start poses for the robot
        # Randomize start position of object
        object_xpos = self.initial_gripper_xpos[:2]
        while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
            object_xpos = self.sim.data.get_body_xpos("table0")[:2] + self.np_random.uniform(
                -self.obj_range, self.obj_range, size=2)
        object_qpos = self.sim.data.get_joint_qpos(self.obj_name + ":joint")
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        # Random rotation around z axis
        object_qpos[3] = self.np_random.rand()
        object_qpos[6] = self.np_random.rand()
        object_qpos[3:7] /= np.linalg.norm(object_qpos[3:7])
        self.sim.data.set_joint_qpos(self.obj_name + ":joint", object_qpos)
        self.sim.forward()
        return True
