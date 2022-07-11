from typing import Dict

from optim.grippers.grippers import Gripper, ParallelJaw, BarrettHand, ShadowHand


def get_gripper(info: Dict) -> Gripper:
    gripper = info["gripper_info"]["type"]
    if gripper == "ParallelJaw":
        return ParallelJaw(info)
    elif gripper == "BarrettHand":
        return BarrettHand(info)
    elif gripper == "ShadowHand":
        return ShadowHand(info)
    raise RuntimeError(f"Gripper {gripper} not supported")
