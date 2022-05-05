"""FlatPJCylinder environment module."""
from pathlib import Path

from gym import utils
import numpy as np

from envs.parallel_jaw.flat_base import FlatPJBase
from envs.rotations import quat_mul, axisangle2quat

MODEL_XML_PATH = str(Path("pj", "flat_pj_cylinder.xml"))


class FlatPJCylinder(FlatPJBase, utils.EzPickle):
    """FlatPJCylinder environment class."""

    def __init__(self, p_rot: float = 0.75, object_size_range: float = 0):
        """Initialize a parallel jaw cylinder environment.

        Args:
            p_rot: Ratio of laying cylinders in the episodes.
            object_size_range: Optional range to enlarge/shrink object sizes.
        """
        FlatPJBase.__init__(self,
                            object_name="cylinder",
                            model_xml_path=MODEL_XML_PATH,
                            object_size_range=object_size_range)
        self.p_rot = p_rot
        utils.EzPickle.__init__(self, p_rot=p_rot, object_size_range=object_size_range)

    def _sample_object_pose(self) -> np.ndarray:
        object_pose = super()._sample_object_pose()
        # Random rotation around z axis
        if self.np_random.rand() < self.p_rot:
            y_rot = axisangle2quat(0, 1, 0, np.pi / 2)
            z_rot = axisangle2quat(0, 0, 1, self.np_random.rand() * np.pi)
            object_rot = quat_mul(z_rot, y_rot)
        else:
            object_rot = axisangle2quat(0, 0, 1, self.np_random.rand() * np.pi)
        object_pose[3:7] = object_rot
        return object_pose
