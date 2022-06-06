"""FlatSHAll environment module."""
import random
from typing import Optional
from pathlib import Path

from gym import utils
import numpy as np

from envs.shadow_hand.flat_base import FlatSHBase
from envs.rotations import quat_mul, axisangle2quat

MODEL_XML_PATH = str(Path("ShadowHand", "flat_all.xml"))


class FlatSHAll(FlatSHBase, utils.EzPickle):
    """FlatSHAll environment class."""

    def __init__(self,
                 n_eigengrasps: Optional[int] = None,
                 object_size_multiplier: float = 1.,
                 object_size_range: float = 0.):
        """Initialize a flat ShadowHand environment with all possible objects.

        Args:
            n_eigengrasps: Number of eigengrasps to use.
            object_size_multiplier: Optional multiplier to change object sizes by a fixed amount.
            object_size_range: Optional range to randomly enlarge/shrink object sizes.
        """
        FlatSHBase.__init__(self,
                            object_name="cube",
                            model_xml_path=MODEL_XML_PATH,
                            n_eigengrasps=n_eigengrasps,
                            object_size_multiplier=object_size_multiplier,
                            object_size_range=object_size_range)
        utils.EzPickle.__init__(self,
                                n_eigengrasps=n_eigengrasps,
                                object_size_multiplier=object_size_multiplier,
                                object_size_range=object_size_range)

    def _env_setup(self, initial_qpos: np.ndarray):
        self.object_name = random.choice(("cube", "cylinder", "sphere", "mesh"))
        super()._env_setup(initial_qpos)

    def _sample_object_pose(self) -> np.ndarray:
        object_pose = self.sim.data.get_joint_qpos(self.object_name + ":joint")
        object_pose[:2] = self.sim.data.get_body_xpos("table0")[:2]
        object_pose[2] = self.height_offset
        if self.object_name == "cube":
            object_rot = self.np_random.rand(4)
            object_rot[1:3] = 0
            object_rot /= np.linalg.norm(object_rot)
        elif self.object_name == "sphere":
            object_rot = np.array([1, 0, 0, 0])
        elif self.object_name in ("cylinder", "mesh"):
            if self.np_random.rand() < 0.5:
                rot_y = axisangle2quat(0, 1, 0, np.pi / 2)
                rot_z = axisangle2quat(0, 0, 1, self.np_random.rand() * np.pi)
                object_rot = quat_mul(rot_z, rot_y)
            else:
                object_rot = axisangle2quat(0, 0, 1, self.np_random.rand() * np.pi)
        else:
            raise RuntimeError(f"Object {self.object_name} not supported")
        object_pose[3:7] = object_rot
        return object_pose
