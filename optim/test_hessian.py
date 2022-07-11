from pathlib import Path
import json
from venv import create

import numpy as np
import jax.numpy as jnp

from envs.rotations import mat2quat
from optim.objective import create_objective_2, create_objective_3
from optim.grippers import get_gripper
from optim.geometry import get_object
from optim.grippers.kinematics.parallel_jaw import _kinematics_right


def main():
    jnp.set_printoptions(precision=3, suppress=True)
    path = Path(__file__).parent / "contact_info_cube.json"
    with open(path, "r") as f:
        info = json.load(f)

    filtered_con_info = [None, None]
    for con_info in info["contact_info"]:
        if con_info["geom1"] == "robot0:r_gripper_finger_link":
            filtered_con_info[0] = con_info
        elif con_info["geom1"] == "robot0:l_gripper_finger_link":
            filtered_con_info[1] = con_info
    assert any([i is not None for i in filtered_con_info])
    info["contact_info"] = filtered_con_info

    # Change contact info from con_pts to kinematics for PJ gripper
    nstates = len(info["gripper_info"]["state"])
    if nstates == 2:
        nstates = 1
    xinit = np.zeros(7 + 2 * nstates)
    xinit[:3] = info["gripper_info"]["pos"]
    xinit[3:7] = mat2quat(np.array(info["gripper_info"]["orient"]))
    xinit[7:7 + nstates] = np.mean(np.array(info["gripper_info"]["state"]))
    # Convert action in [-1, 1] to angle in [0., 0.05]
    xinit[7 + nstates:] = (info["gripper_info"]["next_state"][0] + 1) / 2 * 0.05

    gripper = get_gripper(info)
    kinematics = gripper.create_full_kinematics(gripper.con_links, None)
    positions = kinematics(gripper.state)
    for con_info, pos in zip(info["contact_info"], positions):
        con_info["pos"] = pos
    obj = get_object(info)

    objective, hessian = create_objective_2(gripper, obj.planes[obj.contact_mapping[0]],
                                            obj.planes[obj.contact_mapping[1]])
    lagrangian, grad, hessian = create_objective_3(gripper, obj.planes[obj.contact_mapping[0]])
    xinit = np.append(xinit, 1)
    print(xinit)
    _hessian = np.array(hessian(xinit))
    print(_hessian)


if __name__ == "__main__":
    main()

# jacobian(x->gradient(f,x),xinit)
