"""UnevenBarrettMesh environment module."""
from typing import Optional
from pathlib import Path

from gym import utils

from envs.barrett_hand.uneven_base import UnevenBarrettBase

MODEL_XML_PATH = str(Path("BarrettHand", "uneven_mesh.xml"))


class UnevenBarrettMesh(UnevenBarrettBase, utils.EzPickle):
    """UnevenBarrettMesh environment class."""

    def __init__(self, n_eigengrasps: Optional[int] = None, object_size_multiplier: float = 1.):
        """Initialize a BarrettHand mesh environment with uneven ground.

        Args:
            n_eigengrasps: Number of eigengrasps to use.
            object_size_multiplier: Optional multiplier to change object sizes by a fixed amount.
        """
        UnevenBarrettBase.__init__(self,
                                   object_name="mesh",
                                   model_xml_path=MODEL_XML_PATH,
                                   n_eigengrasps=n_eigengrasps,
                                   object_size_multiplier=object_size_multiplier)
        utils.EzPickle.__init__(self,
                                n_eigengrasps=n_eigengrasps,
                                object_size_multiplier=object_size_multiplier)
