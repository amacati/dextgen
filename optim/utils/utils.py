import json
from typing import Dict

import numpy as np

from optim.rotations import mat2quat


def load_info(path):
    with open(path, "r") as f:
        info = json.load(f)
    return filter_info(info)


def filter_info(info):
    if info["gripper_info"]["type"] == "ParallelJaw":
        return _filter_pj_info(info)
    elif info["gripper_info"]["type"] == "BarrettHand":
        return _filter_bh_info(info)
    else:
        raise KeyError(f"Gripper {info['gripper_info']['type']} not supported!")


def check_grasp(info: Dict) -> bool:
    assert info["gripper_info"]["type"] == "ParallelJaw", "Only ParallelJaw gripper supported"
    links = [con_pt["geom1"] for con_pt in info["contact_info"]]
    if "robot0:r_gripper_finger_link" in links and "robot0:l_gripper_finger_link" in links:
        return True
    return False


def _filter_pj_info(info):
    filtered_con_info = [None, None]
    for con_info in info["contact_info"]:
        if con_info["geom1"] == "robot0:r_gripper_finger_link":
            filtered_con_info[0] = con_info
        elif con_info["geom1"] == "robot0:l_gripper_finger_link":
            filtered_con_info[1] = con_info
    assert any([i is not None for i in filtered_con_info])
    info["contact_info"] = filtered_con_info
    nstates = 1
    orient = np.array(info["gripper_info"]["orient"])
    info["gripper_info"]["orient"] = orient
    info["gripper_info"]["state"] = np.array([np.mean(np.array(info["gripper_info"]["state"]))])
    # Convert action in [-1, 1] to angle in [0., 0.05]
    next_action = (info["gripper_info"]["next_state"] + 1) / 2 * 0.05
    info["gripper_info"]["next_state"] = next_action
    xinit = np.zeros(7 + 2 * nstates)
    xinit[:3] = info["gripper_info"]["pos"]
    xinit[3:7] = mat2quat(info["gripper_info"]["orient"])
    xinit[7:7 + nstates] = info["gripper_info"]["state"]
    xinit[7 + nstates:] = info["gripper_info"]["next_state"]
    return xinit, info


def _filter_bh_info(info):
    nstates = 4
    naction = 4
    xinit = np.zeros(7 + nstates + naction)
    xinit[:3] = info["gripper_info"]["pos"]
    xinit[3:7] = mat2quat(np.array(info["gripper_info"]["orient"]))
    joints = np.array(info["gripper_info"]["state"])
    theta, f1m, f2m, f3m = (joints[0] + joints[3]) / 2, joints[1], joints[4], joints[6]
    xinit[7:7 + nstates] = [theta, f1m, f2m, f3m]
    xinit[-naction:] = info["gripper_info"]["next_state"]
    return xinit, info
