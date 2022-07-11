from typing import Dict

import numpy as np

from optim.grippers.grippers import Gripper
from optim.geometry.base import Geometry
from optim.geometry.normals import create_plane_normals
from optim.constraints import create_plane_constraints


class Cube(Geometry):

    def __init__(self, info: Dict):
        super().__init__(info)
        # Define the 6 sides of the cube as a plane with four border planes at the edges
        # Plane definition:
        # Surface plane origin, surface plane normal
        # Surface border0 origin, surface border0 normal ...
        ex, ey, ez = np.array([1., 0, 0]), np.array([0, 1., 0]), np.array([0, 0, 1.])
        sx, sy, sz = self.size
        plane0 = np.array([[ex * sx, ex], [[sx, sy, 0], ey], [[sx, -sy, 0], -ey], [[sx, 0, sz], ez],
                           [[sx, 0, -sz], -ez]])
        plane1 = np.array([[-ex * sx, -ex], [[-sx, sy, 0], ey], [[-sx, -sy, 0], -ey],
                           [[-sx, 0, sz], ez], [[-sx, 0, -sz], -ez]])
        plane2 = np.array([[ey * sy, ey], [[sx, sy, 0], ex], [[-sx, sy, 0], -ex], [[0, sy, sz],
                                                                                   -ez],
                           [[0, sy, -sz], ez]])
        plane3 = np.array([[-ey * sy, -ey], [[sx, -sy, 0], ex], [[-sx, -sy, 0], -ex],
                           [[0, -sy, sz], -ez], [[0, -sy, -sz], ez]])
        plane4 = np.array([[ez * sz, ez], [[sx, 0, sz], ex], [[-sx, 0, sz], -ex], [[0, sy, sz], ey],
                           [[0, -sy, sz], -ey]])
        plane5 = np.array([[-ez * sz, -ez], [[sx, 0, -sz], ex], [[-sx, 0, -sz], -ex],
                           [[0, sy, -sz], ey], [[0, -sy, -sz], -ey]])
        self.planes = np.array([plane0, plane1, plane2, plane3, plane4, plane5])
        # Rotate and translate planes into correct object pose
        for plane in self.planes:
            for origin_normal in plane:
                for i in range(2):
                    origin_normal[i] = self.orient_mat @ origin_normal[i]  # Rotate all vectors
                    if i == 0:
                        origin_normal[i] += self.pos  # Add position offset to origin points only
        self.contact_mapping = self._contact_mapping()

    def _contact_mapping(self):
        # Calculate the distance of all contact points to all cube planes. Map the contact point to
        # the plane with the smallest distance
        contact_mapping = {}
        for idx, con_pt in enumerate(self.con_pts):
            dst = [np.linalg.norm(con_pt["pos"] - plane[0][0]) for plane in self.planes]
            contact_mapping[idx] = np.argmin(np.abs(np.array(dst)))
        return contact_mapping

    def create_normals(self):
        n = np.array([self.planes[self.contact_mapping[i]][0, 1] for i in range(len(self.con_pts))])
        return create_plane_normals(n)

    def create_surface_constraints(self, gripper: Gripper, opt):
        for idx, con_pt in enumerate(self.con_pts):
            gripper_link = con_pt["geom1"]
            kinematics = gripper.create_kinematics(gripper_link, con_pt)
            plane = self.planes[self.contact_mapping[idx]]
            plane_eq_cnst, plane_ineq_cnsts = create_plane_constraints(kinematics, plane)
            opt.add_equality_constraint(plane_eq_cnst, self.EQ_CNST_TOL)
            # opt.add_inequality_mconstraint(plane_ineq_cnsts, np.ones(4) * self.INEQ_CNST_TOL)
