from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from optim.utils.rotations import mat2quat


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
