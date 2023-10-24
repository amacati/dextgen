"""Utility module."""
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod, abstractproperty
from typing import Optional


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
        return self._run.dir
