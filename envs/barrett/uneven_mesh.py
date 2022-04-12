from typing import Optional
from pathlib import Path

from gym import utils

from envs.barrett.uneven_base import UnevenBarrettBase

MODEL_XML_PATH = str(Path("barrett", "uneven_barrett_mesh.xml"))


class UnevenBarrettMesh(UnevenBarrettBase, utils.EzPickle):

    def __init__(self, n_eigengrasps: Optional[int] = None):
        UnevenBarrettBase.__init__(self,
                                   object_name="mesh",
                                   model_xml_path=MODEL_XML_PATH,
                                   n_eigengrasps=n_eigengrasps)
        utils.EzPickle.__init__(self, n_eigengrasps=n_eigengrasps)
