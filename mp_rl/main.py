import argparse
import logging
from pathlib import Path
import yaml
import torch
import torch.multiprocessing as mp
from fetch_her import fetch_her
from fetch_reach import fetch_reach
from lunar_lander import lunar_lander
from utils import init_process

def main(args):
    size = args.nprocesses
    processes = []
    mp.set_start_method("spawn")
    path = Path(__file__).resolve().parent / "config" / "experiment_config.yaml"
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, train_fn[args.e], config))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    logger.info("Processes joined, training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("e", help="Experiment type that should be run", choices=["lunar", "fetch", "fetch_her"])
    parser.add_argument('--nprocesses', help='Number of worker threads for sample generation',
                        default=8, type=int)
    parser.add_argument('--loglvl', help="Logger levels", choices=["DEBUG", "INFO", "WARN", "ERROR"],
                        default="INFO")
    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    loglvls = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARN": logging.WARN, "ERROR": logging.ERROR}
    logging.basicConfig()
    logging.getLogger().setLevel(loglvls[args.loglvl])
    logger.info("Main process startup")
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda is not available, ddp training not possible!")
    train_fn = {"lunar": lunar_lander.train, "fetch": fetch_reach.train, "fetch_her": fetch_her.train}
    main(args)
