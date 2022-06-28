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

    logger.info("Optimizing contact points")
    opt_config = optimize(info)
    return opt_config


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    main()
