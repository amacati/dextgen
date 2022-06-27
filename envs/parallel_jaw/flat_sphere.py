"""FlatPJSphere environment module."""
from pathlib import Path

from gym import utils

from envs.parallel_jaw.flat_base import FlatPJBase

MODEL_XML_PATH = str(Path("PJ", "flat_sphere.xml"))


class FlatPJSphere(FlatPJBase, utils.EzPickle):
    """FlatPJSphere environment class."""

    def __init__(self, object_size_multiplier: float = 1., object_size_range: float = 0.):
        """Initialize a parallel jaw sphere environment.

        Args:
            object_size_multiplier: Optional multiplier to change object sizes by a fixed amount.
            object_size_range: Optional range to randomly enlarge/shrink object sizes.
        """
        FlatPJBase.__init__(self,
                            object_name="sphere",
                            model_xml_path=MODEL_XML_PATH,
                            object_size_multiplier=object_size_multiplier,
                            object_size_range=object_size_range)
        utils.EzPickle.__init__(self,
                                object_size_multiplier=object_size_multiplier,
                                object_size_range=object_size_range)
