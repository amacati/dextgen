from pathlib import Path

from gym import utils

from envs.parallel_jaw.flat_base import FlatPJBase

MODEL_XML_PATH = str(Path("pj", "flat_pj_sphere.xml"))


class FlatPJSphere(FlatPJBase, utils.EzPickle):

    def __init__(self, object_size_range: float = 0):
        FlatPJBase.__init__(self,
                            object_name="sphere",
                            model_xml_path=MODEL_XML_PATH,
                            object_size_range=object_size_range)
        utils.EzPickle.__init__(self, object_size_range=object_size_range)
