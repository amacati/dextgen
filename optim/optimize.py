from typing import Dict

import numpy as np
import nlopt

from optim.geometry.geometry import get_object
from optim.grippers.grippers import get_gripper


def optimize(info: Dict) -> np.ndarray:
    xinit = np.zeros(6 + len(info["gripper_info"]["state"]))

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

    opt.set_min_objective()

    xopt = opt.optimize(xinit)
    return xopt
