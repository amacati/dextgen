from typing import Dict

from optim.grippers.base_gripper import Gripper
from optim.grippers.parallel_jaw import ParallelJaw
from optim.grippers.barrett_hand import BarrettHand
from optim.grippers.shadow_hand import ShadowHand


def get_gripper(info: Dict) -> Gripper:
    gripper = info["gripper_info"]["type"]
    if gripper == "ParallelJaw":
        return ParallelJaw(info)
    elif gripper == "BarrettHand":
        return BarrettHand(info)
    elif gripper == "ShadowHand":
        return ShadowHand(info)
    raise RuntimeError(f"Gripper {gripper} not supported")