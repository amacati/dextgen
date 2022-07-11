from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from optim.rotations import mat2quat
from optim.constraints import create_max_angle_constraints, create_distance_constraints
from optim.constraints import create_moments_constraints, create_force_constraints


class Geometry(ABC):

    EQ_CNST_TOL = 1e-3
    INEQ_CNST_TOL = 1e-3
    MAX_ALPHA = np.pi / 4
    MAX_ALPHAC = np.cos(MAX_ALPHA)
    MIN_DST = 0.001

    def __init__(self, info: Dict):
        self.com = np.array(info["object_info"]["pos"])
        self.pos = np.array(info["object_info"]["pos"])
        self.orient_mat = np.array(info["object_info"]["orient"])
        self.orient_q = mat2quat(self.orient_mat)
        self.size = np.array(info["object_info"]["size"])
        self.con_pts = info["contact_info"]
        self.con_links = np.array([i["geom1"] for i in info["contact_info"]])
        for con_pt in self.con_pts:  # Numpify contact point arrays
            for key in ("contact_force", "pos", "frame"):
                con_pt[key] = np.array(con_pt[key])

    @abstractmethod
    def create_normals(self):
        ...

    @abstractmethod
    def create_surface_constraints(self, gripper, opt):
        ...

    def create_constraints(self, gripper, opt):
        self.create_surface_constraints(gripper, opt)
        ncon_pts = len(self.con_pts)
        full_kinematics = gripper.create_full_kinematics(self.con_links, self.con_pts)
        grasp_forces = gripper.create_grasp_forces(self.con_links, self.con_pts)
        normals = self.create_normals()
        # angle_ineq_cnsts = create_max_angle_constraints(grasp_forces, normals, self.MAX_ALPHA)
        # opt.add_inequality_mconstraint(angle_ineq_cnsts, np.ones(ncon_pts) * self.INEQ_CNST_TOL)
        # force_eq_cnsts = create_force_constraints(grasp_forces)
        # opt.add_equality_mconstraint(force_eq_cnsts, np.ones(3) * self.EQ_CNST_TOL)
        # moments_eq_cnsts = create_moments_constraints(full_kinematics, grasp_forces, self.com)
        # opt.add_equality_mconstraint(moments_eq_cnsts, np.ones(3) * self.EQ_CNST_TOL)
        # dst_ineq_cnsts = create_distance_constraints(full_kinematics, self.MIN_DST)
        # ndst_cnsts = ncon_pts * (ncon_pts - 1) // 2
        # opt.add_inequality_mconstraint(dst_ineq_cnsts, np.ones(ndst_cnsts) * self.INEQ_CNST_TOL)
