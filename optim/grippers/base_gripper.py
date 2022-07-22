from abc import ABC, abstractmethod, abstractproperty
from typing import Dict, Callable

import numpy as np

from optim.rotations import mat2quat
from optim.constraints import quaternion_cnst


class Gripper(ABC):

    def __init__(self, info: Dict):
        self.pos = np.array(info["gripper_info"]["pos"])
        self.orient_q = mat2quat(np.array(info["gripper_info"]["orient"]))
        self.grip_state = np.array(info["gripper_info"]["state"])
        self.state = np.concatenate((self.pos, self.orient_q, self.grip_state))
        self.con_links = np.array([i["geom1"] for i in info["contact_info"]])

    @abstractmethod
    def create_kinematics(self, con_pt: Dict) -> Callable:
        ...

    @abstractproperty
    def joint_limits(self):
        ...

    def create_constraints(self, opt):
        # Unit quaternion constraint for orientation
        opt.add_equality_constraint(quaternion_cnst)
        low, high = np.tile(self.joint_limits["lower"], 2), np.tile(self.joint_limits["upper"], 2)
        opt.set_lower_bounds(low, begin=7)
        opt.set_upper_bounds(high, begin=7)
