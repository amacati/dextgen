from typing import Dict

from optim.geometry.base import Geometry


class Cylinder(Geometry):

    def __init__(self, info: Dict):
        super().__init__(info)
