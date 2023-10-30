"""FlatPJCube environment module."""
from pathlib import Path

from gymnasium import utils
import numpy as np

from envs.parallel_jaw.flat_base import FlatPJBase

MODEL_XML_PATH = str(Path("PJ", "flat_cube.xml"))


class FlatPJCube(FlatPJBase, utils.EzPickle):
    """FlatPJCube environment class."""

    def __init__(self, object_size_multiplier: float = 1., object_size_range: float = 0.):
        """Initialize a parallel jaw cube environment.

        Args:
            object_size_multiplier: Optional multiplier to change object sizes by a fixed amount.
            object_size_range: Optional range to randomly enlarge/shrink object sizes.
        """
        FlatPJBase.__init__(self,
                            object_name="cube",
                            model_xml_path=MODEL_XML_PATH,
                            object_size_multiplier=object_size_multiplier,
                            object_size_range=object_size_range)
        utils.EzPickle.__init__(self,
                                object_size_multiplier=object_size_multiplier,
                                object_size_range=object_size_range)

    def _sample_object_pose(self) -> np.ndarray:
        object_pose = super()._sample_object_pose()
        # Random rotation around z axis
        object_pose[3] = self.np_random.rand()
        object_pose[6] = self.np_random.rand()
        object_pose[3:7] /= np.linalg.norm(object_pose[3:7])
        return object_pose
