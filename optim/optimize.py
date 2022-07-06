from typing import Dict

import numpy as np
import nlopt

from optim.geometry.geometry import get_object
from optim.grippers.grippers import get_gripper
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
    xinit[7 + nstates:] = info["gripper_info"]["next_state"]

    # Initialize augmented lagrange inner optimizer
    localopt = nlopt.opt(nlopt.LD_MMA, len(xinit))
    localopt.set_lower_bounds(-1)
    localopt.set_upper_bounds(1)
    localopt.set_xtol_rel(1e-3)

    # Initialize lagrange optimizer
    opt = nlopt.opt(nlopt.AUGLAG, len(xinit))
    opt.set_xtol_rel(1e-3)
    opt.set_local_optimizer(localopt)

    obj = get_object(info)
    gripper = get_gripper(info)
    obj.create_constraints(gripper, opt)

    ALPHA_MAX = np.pi / 4
    max_angle_t = np.tan(ALPHA_MAX)
    cp_normals = obj.create_normals()
    grasp_forces = gripper.create_grasp_forces(obj.con_links, obj.con_pts)
    objective = create_objective(cp_normals, grasp_forces, max_angle_t)
    opt.set_min_objective(objective)

    xopt = opt.optimize(xinit)
    return xopt
