import multiprocessing
import logging
from typing import Optional
from pathlib import Path
from typing import Callable
import yaml


logger = logging.getLogger(__name__)


class MPEngine:

    def __init__(self, n_processes: int = 4):
        self.experiment_name = None
        self.experiment_callable = None
        self.processes = []
        self.n_processes = n_processes
        self.queue_manager = multiprocessing.Manager()
        self.results = self.queue_manager.Queue()  # Avoid deadlocking on process join due to open Pipes to result_queue
        self.config = self._load_config()

    def run(self) -> multiprocessing.Queue:
        logger.debug("Engine status check")
        self._check_engine_status()
        logger.debug("Engine status check passed")
        logger.debug("Processes spawn startup")
        for _ in range(self.n_processes):
            p = multiprocessing.Process(target=self.experiment_callable, 
                                        args=(self.config[self.experiment_name], self.results))
            self.processes.append(p)
        logger.debug("Processes spawn complete")
        logger.debug("Processes startup")
        for p in self.processes:
            p.start()
        logger.debug("Processes startup complete")
        logger.info("Parallel run started")
        for p in self.processes:
            p.join()
        logger.debug("Processes join successful, run complete")
        self.processes = []
        return self.results
    
    def register_experiment(self, experiment: Callable, config_path: Optional[Path] = None):
        name = getattr(experiment, '__name__', 'Unknown')
        if name == 'Unknown':
            raise RuntimeError("Experiment callable needs to have a name (__name__ missing)")
        if config_path is not None:
            self.config = self._load_config(config_path)
        if name not in self.config.keys():
            logger.error("Tried to register an experiment without config available")
            raise RuntimeError("Tried to register an experiment without config available")
        self.experiment_name = name
        self.experiment_callable = experiment
        logger.debug("Successful experiment register")
    
    def _check_engine_status(self):
        if len(self.processes) != 0:
            logger.error("Engine status check failed")
            raise RuntimeError("Process queue not empty at engine start")
        if self.experiment_name is None or self.experiment_callable is None:
            logger.error("Engine status check failed")
            raise RuntimeError("No experiment loaded into the engine prior to execution")

    @staticmethod
    def _load_config(path: Optional[Path] = None) -> dict:
        if path is None:
            path = Path(__file__).resolve().parents[1] / "config" / "experiment_config.yaml"
        logger.debug(f"Loading config from path {path}")
        with open(path, "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        return config