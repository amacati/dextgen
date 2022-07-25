from typing import Dict, Callable

import jax.numpy as jnp
from optim.core.optimizer import Optimizer

from optim.grippers.base_gripper import Gripper
from optim.grippers.kinematics.parallel_jaw import PJ_JOINT_LIMITS, kin_pj_right, kin_pj_left


class ParallelJaw(Gripper):

    KP = 30_000
    LINKS = ("robot0:r_gripper_finger_link", "robot0:l_gripper_finger_link")

    def __init__(self, info: Dict):
        super().__init__(info)

    @property
    def joint_limits(self):
        return PJ_JOINT_LIMITS

    def create_kinematics(self, link, con_pt: Dict) -> Callable:
        assert link[7] in ["r", "l"]
        kin_finger = kin_pj_right if link[7] == "r" else kin_pj_left

        def kinematics(x):
            return kin_finger(x)[:3, 3]

        return kinematics

    def create_grasp_force(self, link):
        assert link in self.LINKS
        kin_finger = kin_pj_right if link[0] == "r" else kin_pj_left
        link_axis = 1 if link == self.LINKS[0] else -1  # left gripper moves along -y axis
        xidx = -1 if link == self.LINKS[0] else -2

        def grasp_force(x):
            frame = kin_finger(x)
            f = frame[:3, 1] * link_axis * self.KP * (-x[xidx])
            return jnp.array([*f, 0, 0, 0])  # Wrench with zero torque elements

        return grasp_force

    def create_gripper_constraints(self, opt: Optimizer):

        def pj_finger_constraint(x) -> float:
            return (x[-1] - x[-2])**2

        opt.add_equality_constraint(pj_finger_constraint)
