"""Transformations module for creating homogeneous transformation matrices compatible with JAX."""
import jax.numpy as jnp

from optim.utils.rotations import quat2mat


def tf_matrix(v: jnp.ndarray) -> jnp.ndarray:
    """Construct a homogeneous transformation matrix from a position and euler angles.

    Args:
        v: Array containing the position and euler angles.

    Returns:
        The transformation matrix.
    """
    assert v.shape == (6,), f"Expected shape to be (6, ), got {v.shape}"
    x, y, z, a, b, g = v
    ca = jnp.cos(a)
    sa = jnp.sin(a)
    cb = jnp.cos(b)
    sb = jnp.sin(b)
    cg = jnp.cos(g)
    sg = jnp.sin(g)
    tf = jnp.array([[cb * cg, sa * sb * cg - ca * sg, ca * sb * cg + sa * sg, x],
                    [cb * sg, sa * sb * sg + ca * cg, ca * sb * sg - sa * cg, y],
                    [-sb, sa * cb, ca * cb, z], [0, 0, 0, 1]])
    return tf


def tf_matrix_q(v: jnp.ndarray) -> jnp.ndarray:
    """Construct a homogeneous transformation matrix from a position and a quaterion.

    Args:
        v: Array containing the position and quaternion.

    Returns:
        The transformation matrix.
    """
    assert v.shape == (7,)
    q = v[3:]
    q = q / jnp.linalg.norm(q)
    rot = quat2mat(q)
    tf = jnp.array([[rot[0, 0], rot[0, 1], rot[0, 2],
                     v[0]], [rot[1, 0], rot[1, 1], rot[1, 2], v[1]],
                    [rot[2, 0], rot[2, 1], rot[2, 2], v[2]], [0, 0, 0, 1]])
    return tf


def zrot_matrix(theta: jnp.ndarray) -> jnp.ndarray:
    """Create a homogeneous z-rotation transformation matrix.

    Args:
        theta: Angle of the desired z-rotation.
    """
    return jnp.array([[jnp.cos(theta), -jnp.sin(theta), 0, 0],
                      [jnp.sin(theta), jnp.cos(theta), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
