import json
from pathlib import Path
import logging

from optim.optimize import optimize

logger = logging.getLogger(__name__)


def main():
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

    logger.info("Optimizing contact points")
    opt_config = optimize(info)
    return opt_config


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    main()
