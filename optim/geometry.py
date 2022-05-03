import jax.numpy as jnp
from jax import jit
import numpy as np
from jax.experimental import host_callback as hcb


def generate_sphere_normals(com):

    @jit
    def cp_normals(x):
        cps = x.reshape(-1, 6)[:, :3]
        distances = cps - com
        return distances / jnp.linalg.norm(distances, axis=1, keepdims=True)

    return cp_normals


def generate_sphere_normal(idx, com):

    @jit
    def cp_normal(x):
        cp = x[idx * 6:idx * 6 + 3]
        distance = cp - com
        return distance / jnp.linalg.norm(distance)

    return cp_normal


def generate_plane_normal(normal):

    @jit
    def plane_normal(*_):
        return normal

    return plane_normal


def generate_plane_normals(normals):

    @jit
    def plane_normal(*_):
        return normals

    return plane_normal


def generate_cylinder_normal(idx, com, cylinder_axis, side):
    cylinder_axis / jnp.linalg.norm(cylinder_axis)

    if side == "top":
        return lambda _: cylinder_axis
    elif side == "bottom":
        return lambda _: -cylinder_axis

    @jit
    def cylinder_normal(x):
        cp = x[idx * 6:idx * 6 + 3]
        distance = cp - com
        normal = distance - jnp.dot(distance, cylinder_axis) * cylinder_axis
        return normal / jnp.linalg.norm(normal)

    return cylinder_normal


def generate_cylinder_normals(com, cylinder_axis, sides):
    assert cylinder_axis.shape == (3,)
    cylinder_axis = cylinder_axis / np.linalg.norm(cylinder_axis)
    _cp_lat_idx = np.array([side == "lat" for side in sides])
    normals = jnp.zeros((len(sides), 3))

    for idx, side in enumerate(sides):
        if side == "top":
            normals = normals.at[idx, :].set(cylinder_axis)
        elif side == "bottom":
            normals = normals.at[idx, :].set(-cylinder_axis)

    @jit
    def cylinder_normals(x):
        cps = x.reshape(-1, 6)[_cp_lat_idx, :3]
        distance = cps - com
        rejection = distance - jnp.sum(distance * cylinder_axis, axis=1) * cylinder_axis
        _normals = normals.at[_cp_lat_idx, :].set(rejection /
                                                  jnp.linalg.norm(rejection, axis=1, keepdims=True))
        return _normals

    return cylinder_normals
