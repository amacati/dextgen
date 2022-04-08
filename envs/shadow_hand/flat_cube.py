from pathlib import Path
from typing import Optional

from gym import utils
import numpy as np

from envs.shadow_hand.flat_base import FlatSHBase

MODEL_XML_PATH = str(Path("sh", "flat_sh_cube.xml"))


class FlatSHCube(FlatSHBase, utils.EzPickle):

    def __init__(self, n_eigengrasps: Optional[int] = None, object_size_range: float = 0):
        FlatSHBase.__init__(self,
                            object_name="cube",
                            model_xml_path=MODEL_XML_PATH,
                            n_eigengrasps=n_eigengrasps,
                            object_size_range=object_size_range)
        utils.EzPickle.__init__(self,
                                n_eigengrasps=n_eigengrasps,
                                object_size_range=object_size_range)

    def _sample_object_pose(self) -> np.ndarray:
        object_pose = super()._sample_object_pose()
        # Random rotation around z axis
        object_pose[3] = self.np_random.rand()
        object_pose[6] = self.np_random.rand()
        object_pose[3:7] /= np.linalg.norm(object_pose[3:7])
        return object_pose
