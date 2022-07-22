from typing import Dict, Callable

from optim.grippers.base_gripper import Gripper
from optim.grippers.kinematics.barrett_hand import BH_JOINT_LIMITS


class BarrettHand(Gripper):

    def __init__(self, info: Dict):
        super().__init__(info)

    @property
    def joint_limits(self):
        return BH_JOINT_LIMITS

    def create_kinematics(self, link, con_pt: Dict) -> Callable:
        raise NotImplementedError

    def create_full_kinematics(self, links, _) -> Callable:
        raise NotImplementedError

    def create_full_frames(self, links, _) -> Callable:
        raise NotImplementedError

    def create_grasp_force(self, link):
        raise NotImplementedError

    def create_grasp_forces(self, links, _):
        raise NotImplementedError
