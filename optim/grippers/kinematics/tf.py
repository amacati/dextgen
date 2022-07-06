import jax.numpy as jnp
from jax import jit

from optim.geometry.rotations import quat2mat


@jit
def tf_matrix(v):
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


@jit
def tf_matrix_q(v):
    assert v.shape == (7,)
    x, y, z = v[:3]
    q = v[3:]
    q = q / jnp.linalg.norm(q)
    rot = quat2mat(*q)
    tf = jnp.array([[rot[0, 0], rot[0, 1], rot[0, 2], x], [rot[1, 0], rot[1, 1], rot[1, 2], y],
                    [rot[2, 0], rot[2, 1], rot[2, 2], z], [0, 0, 0, 1]])
    return tf


@jit
def zrot_matrix(theta):
    return jnp.array([[jnp.cos(theta), -jnp.sin(theta), 0, 0],
                      [jnp.sin(theta), jnp.cos(theta), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
