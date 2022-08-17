from __future__ import annotations
from typing import Dict

from optim.grippers.parallel_jaw import ParallelJaw
from optim.grippers.barrett_hand import BarrettHand
from optim.grippers.shadow_hand import ShadowHand
from optim.utils.utils import import_guard

if import_guard():
    from optim.grippers.base_gripper import Gripper  # noqa: TC001, is guarded


def get_gripper(info: Dict) -> Gripper:
    """Create a gripper depending on the information in the contact dict.

    Args:
        info: The contact dict.

    Returns:
        The corresponding gripper object.
    """
    gripper = info["gripper_info"]["type"]
    if gripper == "ParallelJaw":
        return ParallelJaw(info)
    elif gripper == "BarrettHand":
        return BarrettHand(info)
    elif gripper == "ShadowHand":
        return ShadowHand(info)
    raise RuntimeError(f"Gripper {gripper} not supported")
