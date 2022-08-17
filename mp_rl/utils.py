"""Utility module."""
from typing import TYPE_CHECKING


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
