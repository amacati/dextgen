from typing import Optional
from pathlib import Path

from gym import utils

from envs.shadow_hand.uneven_base import UnevenSHBase

MODEL_XML_PATH = str(Path("sh", "uneven_sh_mesh.xml"))


class UnevenSHMesh(UnevenSHBase, utils.EzPickle):

    def __init__(self, n_eigengrasps: Optional[int] = None):
        UnevenSHBase.__init__(self,
                              object_name="mesh",
                              model_xml_path=MODEL_XML_PATH,
                              n_eigengrasps=n_eigengrasps)
        utils.EzPickle.__init__(self, n_eigengrasps=n_eigengrasps)
