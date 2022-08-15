"""Rotations module to extend the `envs.rotations` module by JAX differentiable conversions."""
from typing import Union
import numpy as np
import jax.numpy as jnp


def quat2mat(q: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
    """Convert a single quaternion to a rotation matrix.

    Reimplementation of `envs.rotations` to make it compatible with JAX automatic differentiation.

    Args:
        q: Quaternion.

    Returns:
        The rotation matrix as JAX array.
    """
    r11 = 1 - 2. * (q[2]**2 + q[3]**2)
    r12 = 2. * (q[1] * q[2] - q[0] * q[3])
    r13 = 2. * (q[1] * q[3] + q[0] * q[2])
    r21 = 2. * (q[1] * q[2] + q[0] * q[3])
    r22 = 1 - 2. * (q[1]**2 + q[3]**2)
    r23 = 2. * (q[2] * q[3] - q[0] * q[1])
    r31 = 2. * (q[1] * q[3] - q[0] * q[2])
    r32 = 2. * (q[2] * q[3] + q[0] * q[1])
    r33 = 1 - 2. * (q[1]**2 + q[2]**2)
    return jnp.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
