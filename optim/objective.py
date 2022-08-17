"""Objective function factory module."""
from typing import Callable, Tuple

import numpy as np
import jax.numpy as jnp

from optim.grippers.kinematics.parallel_jaw import kin_pj_right, kin_pj_left


def create_cube_objective(xinit: Tuple[np.ndarray, jnp.ndarray], com: np.ndarray) -> Callable:
    """Create an objective function for the cube optimization.

    Args:
        xinit: Optimization variable on initialization.
        com: Cube center of mass.

    Returns:
        The objective function callable.
    """

    def objective(x: jnp.ndarray) -> float:
        """Calculate the value of the objective function at the current configuration.

        Args:
            x: The current configuration.

        Returns:
            The value of the objective.
        """
        regularizer = jnp.sum((x - xinit)**2)
        diff_r, diff_l = kin_pj_right(x)[:3, 3] - com, kin_pj_left(x)[:3, 3] - com
        return regularizer * 1e-4 + jnp.sum(diff_r**2) + jnp.sum(diff_l**2)

    return objective
