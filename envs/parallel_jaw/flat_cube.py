"""FlatPJCube environment module."""
from pathlib import Path

from gym import utils
import numpy as np

from envs.parallel_jaw.flat_base import FlatPJBase

MODEL_XML_PATH = str(Path("pj", "flat_pj_cube.xml"))


class FlatPJCube(FlatPJBase, utils.EzPickle):
    """FlatPJCube environment class."""

    def __init__(self, object_size_range: float = 0):
        """Initialize a parallel jaw cube environment.

        Args:
            object_size_range: Optional range to enlarge/shrink object sizes.
        """
        FlatPJBase.__init__(self,
                            object_name="cube",
                            model_xml_path=MODEL_XML_PATH,
                            object_size_range=object_size_range)
        utils.EzPickle.__init__(self, object_size_range=object_size_range)

    def _sample_object_pose(self) -> np.ndarray:
        object_pose = super()._sample_object_pose()
        # Random rotation around z axis
        object_pose[3] = self.np_random.rand()
        object_pose[6] = self.np_random.rand()
        object_pose[3:7] /= np.linalg.norm(object_pose[3:7])
        return object_pose
