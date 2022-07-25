from typing import Dict

import numpy as np

from optim.grippers.base_gripper import Gripper
from optim.grippers.kinematics.parallel_jaw import kin_pj_full
from optim.geometry.base_geometry import Geometry
from optim.geometry.normals import create_plane_normals
from optim.constraints import create_plane_constraints


class Cube(Geometry):

    def __init__(self, info: Dict, gripper: Gripper):
        super().__init__(info)
        # Define the 6 sides of the cube as a plane with four border planes at the edges
        # Plane definition:
        # Surface plane offset, surface border0 offset, ...
        # Surface plane normal, surface border0 normal, ...
        ex, ey, ez = np.array([1., 0, 0]), np.array([0, 1., 0]), np.array([0, 0, 1.])
        sx, sy, sz = self.size
        plane0_offsets = np.array([ex * sx, [sx, sy, 0], [sx, -sy, 0], [sx, 0, sz], [sx, 0, -sz]])
        plane0_normals = np.array([ex, ey, -ey, ez, -ez])
        plane1_offsets = np.array(
            [-ex * sx, [-sx, sy, 0], [-sx, -sy, 0], [-sx, 0, sz], [-sx, 0, -sz]])
        plane1_normals = np.array([-ex, ey, -ey, ez, -ez])
        plane2_offsets = np.array([ey * sy, [sx, sy, 0], [-sx, sy, 0], [0, sy, sz], [0, sy, -sz]])
        plane2_normals = np.array([ey, ex, -ex, ez, -ez])
        plane3_offsets = np.array(
            [-ey * sy, [sx, -sy, 0], [-sx, -sy, 0], [0, -sy, sz], [0, -sy, -sz]])
        plane3_normals = np.array([-ey, ex, -ex, ez, -ez])
        plane4_offsets = np.array([ez * sz, [sx, 0, sz], [-sx, 0, sz], [0, sy, sz], [0, -sy, sz]])
        plane4_normals = np.array([ez, ex, -ex, ey, -ey])
        plane5_offsets = np.array(
            [-ez * sz, [sx, 0, -sz], [-sx, 0, -sz], [0, sy, -sz], [0, -sy, -sz]])
        plane5_normals = np.array([-ez, ex, -ex, ey, -ey])
        self.plane_offsets = np.array([
            plane0_offsets, plane1_offsets, plane2_offsets, plane3_offsets, plane4_offsets,
            plane5_offsets
        ])
        self.plane_normals = np.array([
            plane0_normals, plane1_normals, plane2_normals, plane3_normals, plane4_normals,
            plane5_normals
        ])
        # Rotate and translate planes into correct object pose
        for i in range(6):
            for j in range(5):
                self.plane_offsets[i, j] = self.orient_mat @ self.plane_offsets[i, j] + self.pos
                self.plane_normals[i, j] = self.orient_mat @ self.plane_normals[i, j]
        self.contact_mapping = self._contact_mapping(gripper)

    def _contact_mapping(self, gripper):
        # Calculate the distance of all contact points to all cube planes. Map the contact point to
        # the plane with the smallest distance
        contact_mapping = {}
        frames = kin_pj_full(gripper.state)
        posr, posl = frames[1][:3, 3], frames[3][:3, 3]
        for idx, con_pt in enumerate(self.con_pts):
            con_pt_pos = posr if con_pt["geom1"] == gripper.LINKS[0] else posl
            dst = [np.linalg.norm(con_pt_pos - pos[0]) for pos in self.plane_offsets]
            contact_mapping[idx] = np.argmin(np.abs(np.array(dst)))
        return contact_mapping

    def create_normals(self):
        n_normals = len(self.con_pts)
        n = np.array([self.plane_normals[self.contact_mapping[i]][0] for i in range(n_normals)])
        return create_plane_normals(n)

    def create_surface_constraints(self, gripper: Gripper, opt):
        for i, con_pt in enumerate(self.con_pts):
            kinematics = gripper.create_kinematics(self.con_links[i], con_pt)
            plane_idx = self.contact_mapping[i]
            offsets, normals = self.plane_offsets[plane_idx], self.plane_normals[plane_idx]
            plane_eq_cnst, plane_ineq_cnsts = create_plane_constraints(kinematics, offsets, normals)
            opt.add_equality_constraint(plane_eq_cnst)
            opt.add_inequality_mconstraint(plane_ineq_cnsts)
