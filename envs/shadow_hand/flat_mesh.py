from pathlib import Path
from typing import Optional

from gym import utils
import numpy as np

from envs.shadow_hand.flat_base import FlatSHBase
from envs.rotations import axisangle2quat, quatmultiply

MODEL_XML_PATH = str(Path("sh", "flat_sh_mesh.xml"))


class FlatSHMesh(FlatSHBase, utils.EzPickle):

    def __init__(self, n_eigengrasps: Optional[int] = None, object_size_range: float = 0):
        FlatSHBase.__init__(self,
                            object_name="mesh",
                            model_xml_path=MODEL_XML_PATH,
                            n_eigengrasps=n_eigengrasps,
                            object_size_range=object_size_range)
        utils.EzPickle.__init__(self,
                                n_eigengrasps=n_eigengrasps,
                                object_size_range=object_size_range)

    def _env_setup(self, initial_qpos: np.ndarray):
        object_pose = self.sim.data.get_joint_qpos(self.object_name + ":joint")
        object_pose[:2] = self.sim.data.get_body_xpos("table0")[:2]
        object_pose[2] = self.height_offset
        # Rotate object
        if self.np_random.rand() < 0.5:
            rot_y = axisangle2quat(0, 1, 0, np.pi / 2)
            rot_z = axisangle2quat(0, 0, 1, self.np_random.rand() * np.pi)
            object_rot = quatmultiply(rot_z, rot_y)
        else:
            object_rot = axisangle2quat(0, 0, 1, self.np_random.rand() * np.pi)
        object_pose[3:7] = object_rot
        self.sim.data.set_joint_qpos(self.object_name + ":joint", object_pose)
        # Remove mesh from initial_qpos to keep the modified joint position
        modified_qpos = initial_qpos.copy()
        modified_qpos.pop("mesh:joint", None)
        super()._env_setup(modified_qpos)
