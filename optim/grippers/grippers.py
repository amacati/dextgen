from abc import ABC, abstractmethod, abstractproperty
from typing import Dict, Callable

from jax import jit
import numpy as np

from optim.grippers.kinematics.tf import tf_matrix
from optim.grippers.kinematics.parallel_jaw import pj_kinematics, PJ_JOINT_LIMITS
from optim.geometry.rotations import mat2quat


class Gripper(ABC):

    POS_CNST = np.ones(3) * 2
    ORIENT_CNST = np.ones(4)

    def __init__(self, info: Dict):
        self.pos = np.array(info["gripper_info"]["pos"])
        self.orient_q = mat2quat(np.array(info["gripper_info"]["orient"]))
        self.grip_state = np.array(info["gripper_info"]["state"])
        self.state = np.concatenate((self.pos, self.orient_q, self.grip_state))

    @abstractmethod
    def create_kinematics(self, con_pt: Dict) -> Callable:
        ...

    @abstractproperty
    def joint_limits(self):
        ...

    def create_constraints(self, opt):
        # Unit quaternion constraint for orientation

        low_cnst = np.concatenate((-self.POS_CNST, -self.ORIENT_CNST, self.joint_limits["lower"]))
        high_cnst = np.concatenate((self.POS_CNST, self.ORIENT_CNST, self.joint_limits["upper"]))
        opt.set_lower_bounds(low_cnst)
        opt.set_upper_bounds(high_cnst)


class ParallelJaw(Gripper):

    def __init__(self, info: Dict):
        super().__init__(info)

    @property
    def joint_limits(self):
        return PJ_JOINT_LIMITS

    def create_kinematics(self, link, con_pt: Dict) -> Callable:
        link_kinematics = pj_kinematics(link)
        dx = con_pt["pos"] - link_kinematics(self.state)[:3, 3]
        tf = tf_matrix(*dx, 0, 0, 0)

        @jit
        def kinematics(x):
            return (link_kinematics(x) @ tf)[0:3, 3]

        return kinematics

    def create_grasp_force(self, link):
        ...


class BarrettHand(Gripper):

    def __init__(self, info: Dict):
        super().__init__(info)


class ShadowHand(Gripper):

    def __init__(self, info: Dict):
        super().__init__(info)


def get_gripper(info: Dict) -> Gripper:
    gripper = info["gripper_info"]["type"]
    if gripper == "ParallelJaw":
        return ParallelJaw(info)
    elif gripper == "BarrettHand":
        return BarrettHand(info)
    elif gripper == "ShadowHand":
        return ShadowHand(info)
    raise RuntimeError(f"Gripper {gripper} not supported")
