"""FlatPJCube environment module."""
from pathlib import Path
from typing import Tuple

from gym import utils
import numpy as np

from envs.parallel_jaw.flat_base import FlatPJBase

MODEL_XML_PATH = str(Path("PJ", "flat_cube.xml"))


class FlatPJCube(FlatPJBase, utils.EzPickle):
    """FlatPJCube environment class."""

    def __init__(
            self,
            p_high_goal: float = 0.5,
            goal_range: Tuple[float, float] = (0.0, 0.3),
    ):
        """Initialize a parallel jaw cube environment.

        Args:
            p_high_goal: Probability of sampling a goal that is above the surface.
            goal_range: Range of object goal heights.
        """
        FlatPJBase.__init__(self,
                            model_xml_path=MODEL_XML_PATH,
                            p_high_goal=p_high_goal,
                            goal_range=goal_range)
        utils.EzPickle.__init__(self, p_high_goal=p_high_goal, goal_range=goal_range)

    def _sample_object_pose(self) -> np.ndarray:
        object_pose = super()._sample_object_pose()
        # Random rotation around z axis
        object_pose[3] = self.np_random.rand()
        object_pose[6] = self.np_random.rand()
        object_pose[3:7] /= np.linalg.norm(object_pose[3:7])
        return object_pose
