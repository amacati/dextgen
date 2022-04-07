from gym import utils

from envs.shadow_hand.uneven_base import UnevenSHBase


class UnevenSHCube(UnevenSHBase, utils.EzPickle):

    def __init__(self, n_eigengrasps: int):
        UnevenSHBase.__init__(self, obj_name="cube", n_eigengrasps=n_eigengrasps)
        utils.EzPickle.__init__(self, n_eigengrasps=n_eigengrasps)
