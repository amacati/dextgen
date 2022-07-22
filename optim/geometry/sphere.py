from typing import Dict

from optim.geometry.base_geometry import Geometry
from optim.grippers import Gripper


class Sphere(Geometry):

    def __init__(self, info: Dict):
        super().__init__(info)

    def create_normals(self):
        raise NotImplementedError

    def create_surface_constraints(self, gripper: Gripper, opt):
        raise NotImplementedError
