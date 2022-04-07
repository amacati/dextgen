from gym import utils
import numpy as np

from envs.parallel_jaw.flat_base import FlatPJBase


class FlatPJCube(FlatPJBase, utils.EzPickle):

    def __init__(self, object_size_range: float = 0):
        FlatPJBase.__init__(self, object_name="cube", object_size_range=object_size_range)
        utils.EzPickle.__init__(self, object_size_range=object_size_range)

    def _sample_object_pose(self) -> np.ndarray:
        object_pose = super()._sample_object_pose()
        # Random rotation around z axis
        object_pose[3] = self.np_random.rand()
        object_pose[6] = self.np_random.rand()
        object_pose[3:7] /= np.linalg.norm(object_pose[3:7])
        return object_pose
