"""Utility module."""
from typing import TYPE_CHECKING, Any
from abc import ABC, abstractmethod, abstractproperty
from typing import Optional
from pathlib import Path


def import_guard() -> bool:
    """Check if type checking is active or sphinx is trying to build the docs.

    Returns:
        True if either type checking is active or sphinx builds the docs, else False.
    """
    if TYPE_CHECKING:
        return True
    try:  # Not unreachable, TYPE_CHECKING deactivated for sphinx docs build
        if __sphinx_build__:
            return True
    except NameError:
        return False


class DummyWandBConfig:

    def __init__(self, config: dict):
        self._config = config

    def __getattr__(self, item):
        if item not in self._config:
            raise AttributeError(f"Config has no attribute {item}.")
        return self._config.get(item)["value"]

    def __setattr__(self, name: str, value: Any):
        if name == "_config":
            super().__setattr__(name, value)
            return
        assert name in self._config, f"Config has no attribute {name}."
        self._config[name]["value"] = value


class Logger(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def log(self, log, step: Optional[int] = None, commit: bool = False):
        raise NotImplementedError

    @abstractproperty
    def path(self):
        raise NotImplementedError


class DummyLogger(Logger):

    def __init__(self):
        super().__init__()

    def log(self, log, step: Optional[int] = None, commit: bool = False):
        pass

    @property
    def path(self):
        raise RuntimeError('No path for dummy logger.')


class WandBLogger(Logger):

    def __init__(self, run):
        super().__init__()
        self._run = run

    def log(self, log, step: Optional[int] = None, commit: bool = False):
        self._run.log(log, step=step, commit=commit)

    @property
    def path(self):
        return Path(self._run.dir)
