"""Utility module."""
from typing import Dict, Tuple, TYPE_CHECKING

import numpy as np

from envs.rotations import mat2quat


def filter_info(info: Dict) -> Tuple[np.ndarray, Dict]:
    """Filter the contact info of a contact information dictionary.

    Dispatch function for general contact info dicts. Since the gripper type is given as a string
    and not as an object, we do not use a proper singledispatch.

    Args:
        info: Contact information dictionary.

    Returns:
        The initial gripper configuration for the optimization and the filtered contact info.

    Raises:
        KeyError: Gripper type is not supported.
    """
    if info["gripper_info"]["type"] == "ParallelJaw":
        return _filter_pj_info(info)
    else:
        raise KeyError(f"Gripper {info['gripper_info']['type']} not supported!")


def check_grasp(info: Dict) -> bool:
    """Check if the current grasp constitutes a valid configuration.

    Args:
        info: Contact information dictionary.

    Returns:
        True if the grasp is an optimization initialization candidate, else False.

    Raises:
        AssertionError: Contact info contains an unsupported gripper type.
    """
    assert info["gripper_info"]["type"] == "ParallelJaw", "Only ParallelJaw gripper supported"
    links = [con_pt["geom1"] for con_pt in info["contact_info"]]
    if "robot0:r_gripper_finger_link" in links and "robot0:l_gripper_finger_link" in links:
        return True
    return False


def _filter_pj_info(info: Dict) -> Tuple[np.ndarray, Dict]:
    """Filter information from the ParallelJaw gripper and extract the initial configuration.

    Convert joint angles to the range of -1, 1, numpify arrays and convert orientation descriptions.

    Args:
        info: Contact information dictionary.

    Returns:
        The initial gripper configuration for the optimization and the converted contact dictionary.
    """
    filtered_con_info = [None, None]
    for con_info in info["contact_info"]:
        if con_info["geom1"] == "robot0:r_gripper_finger_link":
            filtered_con_info[0] = con_info
        elif con_info["geom1"] == "robot0:l_gripper_finger_link":
            filtered_con_info[1] = con_info
    assert any([i is not None for i in filtered_con_info])
    info["contact_info"] = filtered_con_info
    nstates = 2
    orient = np.array(info["gripper_info"]["orient"])
    info["gripper_info"]["orient"] = orient
    info["gripper_info"]["state"] = np.array(info["gripper_info"]["state"])
    # Convert action in [-1, 1] to angle in [0., 0.05]
    next_action = (info["gripper_info"]["next_state"] + 1) / 2 * 0.05
    info["gripper_info"]["next_state"] = next_action
    xinit = np.zeros(7 + nstates)
    xinit[:3] = info["gripper_info"]["pos"]
    xinit[3:7] = mat2quat(info["gripper_info"]["orient"])
    xinit[7:7 + nstates] = info["gripper_info"]["state"]
    return xinit, info


def import_guard() -> bool:
    """Check if type checking is active or sphinx is trying to build the docs.

    Returns:
        True if either type checking is active or sphinx builds the docs, else False.
    """
    if TYPE_CHECKING:
        return True
    try:  # Not unreachable, TYPE_CHECKING deactivated for sphinx docs build
        if __sphinx_build__:
            return True
    except NameError:
        return False
