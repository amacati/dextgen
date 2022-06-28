from jax import jit
import jax.numpy as jnp
import numpy as np


@jit
def mat2quat(mat):
    w = jnp.sqrt(1 + mat[0, 0] + mat[1, 1] + mat[2, 2]) / 2.
    w4 = 4. * w
    x = (mat[2, 1] - mat[1, 2]) / w4
    y = (mat[0, 2] - mat[2, 0]) / w4
    z = (mat[1, 0] - mat[0, 1]) / w4
    return jnp.array([w, x, y, z])


@jit
def quat2mat(w, x, y, z):
    return jnp.array([[1 - 2. * (y**2 + z**2), 2. * (x * y - w * z), 2. * (x * z + w * y)],
                      [2. * (x * y + w * z), 1 - 2. * (x**2 + z**2), 2. * (y * z - w * x)],
                      [2. * (x * z - w * y), 2. * (y * z + w * x), 1 - 2. * (x**2 + y**2)]])
