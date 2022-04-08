from pathlib import Path
from typing import Optional

from gym import utils

from envs.shadow_hand.flat_base import FlatSHBase

MODEL_XML_PATH = str(Path("sh", "flat_sh_sphere.xml"))


class FlatSHSphere(FlatSHBase, utils.EzPickle):

    def __init__(self, n_eigengrasps: Optional[int] = None, object_size_range: float = 0):
        FlatSHBase.__init__(self,
                            object_name="sphere",
                            model_xml_path=MODEL_XML_PATH,
                            n_eigengrasps=n_eigengrasps,
                            object_size_range=object_size_range)
        utils.EzPickle.__init__(self,
                                n_eigengrasps=n_eigengrasps,
                                object_size_range=object_size_range)
