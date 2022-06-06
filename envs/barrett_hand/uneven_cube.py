"""UnevenBarrettCube environment module."""
from pathlib import Path
from typing import Optional

from gym import utils

from envs.barrett_hand.uneven_base import UnevenBarrettBase

MODEL_XML_PATH = str(Path("BarrettHand", "uneven_cube.xml"))


class UnevenBarrettCube(UnevenBarrettBase, utils.EzPickle):
    """UnevenBarrettCube environment class."""

    def __init__(self, n_eigengrasps: Optional[int] = None, object_size_multiplier: float = 1.):
        """Initialize a BarrettHand cube environment with uneven ground.

        Args:
            n_eigengrasps: Number of eigengrasps to use.
            object_size_multiplier: Optional multiplier to change object sizes by a fixed amount.
        """
        UnevenBarrettBase.__init__(self,
                                   object_name="cube",
                                   model_xml_path=MODEL_XML_PATH,
                                   n_eigengrasps=n_eigengrasps,
                                   object_size_multiplier=object_size_multiplier)
        utils.EzPickle.__init__(self,
                                n_eigengrasps=n_eigengrasps,
                                object_size_multiplier=object_size_multiplier)
