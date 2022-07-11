from typing import Dict

from optim.geometry.base import Geometry
from optim.grippers import Gripper
from optim.constraints import create_sphere_constraint


class Sphere(Geometry):

    def __init__(self, info: Dict):
        super().__init__(info)

    def create_normals(self):
        ...

    def create_surface_constraints(self, gripper: Gripper, opt):
        for con_pt in self.con_pts:
            kinematics = gripper.create_kinematics(con_pt)
            sphere_eq_cnst = create_sphere_constraint(kinematics, self.pos, self.size[0])
            opt.add_equality_constraint(sphere_eq_cnst, self.EQ_CNST_TOL)
