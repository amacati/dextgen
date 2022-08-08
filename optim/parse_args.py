"""Argument parser helper functions."""
import argparse
from pathlib import Path
import logging

import yaml

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse arguments for the gym environment and logging levels.

    Returns:
        The parsed arguments as a namespace.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--loglvl',
                        help="Logger levels",
                        choices=["DEBUG", "INFO", "WARN", "ERROR"],
                        default="INFO")
    parser.add_argument("--render",
                        help="Render flag. Only used for testing",
                        choices=["y", "n"],
                        default="y")
    parser.add_argument("--record",
                        help="Record video flag. Only used for testing",
                        choices=["y", "n"],
                        default="n")
    parser.add_argument("--ntests",
                        help="Number of evaluation runs. Only used for testing",
                        default=10,
                        type=int)
    args = parser.parse_args()
    args.env = "FlatPJCube-v0"
    expand_args(args)
    return args


def expand_args(args: argparse.Namespace):
    """Expand the arguments namespace with settings from the main config file.

    Config can be found at './config/experiment_config.yaml'. Each config must be named after their
    gym name.

    Args:
        args: User provided arguments namespace.
    """
    logging.basicConfig(level=logging.INFO)
    path = Path(__file__).parent.parent / "config" / "experiment_config.yaml"
    with open(path, "r") as f:
        config = yaml.load(f, yaml.SafeLoader)

    if "Default" not in config.keys():
        raise KeyError("Config file is missing required entry `Default`!")
    for key, val in config["Default"].items():
        setattr(args, key, val)

    if args.env not in config.keys():
        logger.info(f"No specific config for {args.env} found, using defaults for all settings.")
    else:
        for key, val in config[args.env].items():
            setattr(args, key, val)
