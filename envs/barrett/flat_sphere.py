"""FlatBarrettSphere environment module."""
from pathlib import Path
from typing import Optional

from gym import utils

from envs.barrett.flat_base import FlatBarrettBase

MODEL_XML_PATH = str(Path("barrett", "flat_barrett_sphere.xml"))


class FlatBarrettSphere(FlatBarrettBase, utils.EzPickle):
    """FlatBarrettSphere environment class."""

    def __init__(self,
                 n_eigengrasps: Optional[int] = None,
                 object_size_multiplier: float = 1.,
                 object_size_range: float = 0.):
        """Initialize a BarrettHand sphere environment.

        Args:
            n_eigengrasps: Number of eigengrasps to use.
            object_size_multiplier: Optional multiplier to change object sizes by a fixed amount.
            object_size_range: Optional range to randomly enlarge/shrink object sizes.
        """
        FlatBarrettBase.__init__(self,
                                 object_name="sphere",
                                 model_xml_path=MODEL_XML_PATH,
                                 n_eigengrasps=n_eigengrasps,
                                 object_size_multiplier=object_size_multiplier,
                                 object_size_range=object_size_range)
        utils.EzPickle.__init__(self,
                                n_eigengrasps=n_eigengrasps,
                                object_size_multiplier=object_size_multiplier,
                                object_size_range=object_size_range)
