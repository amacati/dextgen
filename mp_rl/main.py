import argparse
import multiprocessing as mp
import logging
from lunar_lander import lunar_lander


def main(args):
    logging.basicConfig()
    logging.getLogger().setLevel(args.loglvl)
    lunar_lander.lunar_lander_ddpg(args)


if __name__ == "__main__":
    mp.set_start_method('spawn')  # Cuda cannot re-initialize in forked subprocesses
    parser = argparse.ArgumentParser()
    parser.add_argument('--nprocesses', help='Number of worker threads for sample generation',
                        default=8)
    parser.add_argument('--loglvl', help="Logger levels", choices=["DEBUG", "INFO", "WARN", "ERROR"],
                        default="INFO")
    args = parser.parse_args()
    if args.loglvl == "DEBUG":
        args.loglvl = logging.DEBUG
    elif args.loglvl == "INFO":
        args.loglvl = logging.INFO
    elif args.loglvl == "WARN":
        args.loglvl = logging.WARN
    elif args.loglvl == "ERROR":
        args.loglvl = logging.ERROR
    main(args)
