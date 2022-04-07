from gym import utils

from envs.parallel_jaw.flat_base import FlatPJBase


class FlatPJSphere(FlatPJBase, utils.EzPickle):

    def __init__(self, object_size_range: float = 0):
        FlatPJBase.__init__(self, object_name="sphere", object_size_range=object_size_range)
        utils.EzPickle.__init__(self, object_size_range=object_size_range)
