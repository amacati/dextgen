from gym import utils

from envs.shadow_hand.flat_base import FlatSHBase


class FlatSHSphere(FlatSHBase, utils.EzPickle):

    def __init__(self, n_eigengrasps: int, object_size_range: float = 0):
        FlatSHBase.__init__(self,
                            object_name="sphere",
                            n_eigengrasps=n_eigengrasps,
                            object_size_range=object_size_range)
        utils.EzPickle.__init__(self,
                                n_eigengrasps=n_eigengrasps,
                                object_size_range=object_size_range)
