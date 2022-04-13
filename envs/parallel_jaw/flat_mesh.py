from pathlib import Path

from gym import utils
import numpy as np

from envs.parallel_jaw.flat_base import FlatPJBase
from envs.rotations import axisangle2quat, quat_mul

MODEL_XML_PATH = str(Path("pj", "flat_pj_mesh.xml"))


class FlatPJMesh(FlatPJBase, utils.EzPickle):

    def __init__(self, object_size_range: float = 0):
        FlatPJBase.__init__(self,
                            object_name="mesh",
                            model_xml_path=MODEL_XML_PATH,
                            object_size_range=object_size_range)
        utils.EzPickle.__init__(self, object_size_range=object_size_range)

    def _env_setup(self, initial_qpos: np.ndarray):
        object_pose = self.sim.data.get_joint_qpos(self.object_name + ":joint")
        object_pose[:2] = self.sim.data.get_body_xpos("table0")[:2]
        object_pose[2] = self.height_offset
        # Rotate object
        if self.np_random.rand() < 0.5:
            rot_y = axisangle2quat(0, 1, 0, np.pi / 2)
            rot_z = axisangle2quat(0, 0, 1, self.np_random.rand() * np.pi)
            object_rot = quat_mul(rot_z, rot_y)
        else:
            object_rot = axisangle2quat(0, 0, 1, self.np_random.rand() * np.pi)
        object_pose[3:7] = object_rot
        self.sim.data.set_joint_qpos(self.object_name + ":joint", object_pose)
        # Remove mesh from initial_qpos to keep the modified joint position
        modified_qpos = initial_qpos.copy()
        modified_qpos.pop("mesh:joint", None)
        super()._env_setup(modified_qpos)
