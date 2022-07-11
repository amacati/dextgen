from typing import Dict

import numpy as np
import nlopt

from optim.geometry import get_object
from optim.grippers import get_gripper
from optim.objective import create_objective
from envs.rotations import mat2quat


def optimize(info: Dict) -> np.ndarray:
    nstates = len(info["gripper_info"]["state"])
    if nstates == 2:
        nstates = 1
    xinit = np.zeros(7 + 2 * nstates)
    xinit[:3] = info["gripper_info"]["pos"]
    xinit[3:7] = mat2quat(np.array(info["gripper_info"]["orient"]))
    xinit[7:7 + nstates] = np.mean(np.array(info["gripper_info"]["state"]))
    # Convert action in [-1, 1] to angle in [0., 0.05]
    xinit[7 + nstates:] = (info["gripper_info"]["next_state"][0] + 1) / 2 * 0.05

    # Initialize augmented lagrange inner optimizer. Bounds are set in gripper.create_constraints()
    localopt = nlopt.opt(nlopt.LD_MMA, len(xinit))
    localopt.set_xtol_rel(1e-3)

    # Initialize lagrange optimizer
    opt = nlopt.opt(nlopt.AUGLAG, len(xinit))
    opt.set_xtol_rel(1e-3)
    opt.set_local_optimizer(localopt)

    gripper = get_gripper(info)
    kinematics = gripper.create_full_kinematics(gripper.con_links, None)
    positions = kinematics(gripper.state)
    for con_info, pos in zip(info["contact_info"], positions):
        con_info["pos"] = pos
    obj = get_object(info)
    obj.create_constraints(gripper, opt)
    gripper.create_constraints(opt, localopt)

    ALPHA_MAX = np.pi / 4
    max_angle_t = np.tan(ALPHA_MAX)
    cp_normals = obj.create_normals()
    grasp_forces = gripper.create_grasp_forces(obj.con_links, obj.con_pts)
    objective = create_objective(cp_normals, grasp_forces, max_angle_t)
    print(f"Initial force reserve: {objective(xinit, np.array([])):.2f}")
    opt.set_maxeval(1)
    opt.set_min_objective(objective)

    print(xinit)
    xopt = opt.optimize(xinit)
    print(xopt)
    print(f"Force reserve: {opt.last_optimum_value():.2f}")
    return xopt
