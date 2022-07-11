import json
from pathlib import Path
import logging

from jax.config import config
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from optim.optimize import optimize
from envs.rotations import mat2quat
from optim.visualization import visualize_grasp, visualize_gripper
from optim.grippers import get_gripper
from optim.geometry import get_object

logger = logging.getLogger(__name__)


def main():
    jnp.set_printoptions(precision=3, suppress=True)
    path = Path(__file__).parent / "contact_info_cube.json"
    with open(path, "r") as f:
        info = json.load(f)
    logger.info("Loaded contact info")

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

    logger.info("Optimizing contact points")
    opt_config = optimize(info)

    gripper.state = opt_config
    fig = visualize_grasp(obj, gripper, opt_config)
    gripper.state = xinit
    fig = visualize_gripper(gripper, fig, color="k")
    for con_pt in info["contact_info"]:
        pos = con_pt["pos"]
        fig.axes[0].scatter(*pos, color="b")
    plt.show()
    return opt_config


if __name__ == "__main__":
    config.update("jax_debug_nans", True)
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    main()
