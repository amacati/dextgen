import jax.numpy as jnp
from jax import jit
import numpy as np


def create_sphere_normals(full_kinematics, com):

    @jit
    def cp_normals(x):
        distances = full_kinematics(x) - com
        return distances / jnp.linalg.norm(distances, axis=1, keepdims=True)

    return cp_normals


def create_sphere_normal(kinematics, com):

    @jit
    def cp_normal(x):
        distance = kinematics(x) - com
        return distance / jnp.linalg.norm(distance)

    return cp_normal


def create_plane_normal(normal):

    @jit
    def plane_normal(*_):
        return normal

    return plane_normal


def create_plane_normals(normals):

    @jit
    def plane_normal(*_):
        return normals

    return plane_normal


def create_cylinder_normal(kinematics, com, cylinder_axis, side):
    cylinder_axis / jnp.linalg.norm(cylinder_axis)

    if side == "top":

        @jit
        def cylinder_normal(x):
            return cylinder_axis

    elif side == "bottom":

        @jit
        def cylinder_normal(x):
            return -cylinder_axis
    else:

        @jit
        def cylinder_normal(x):
            distance = kinematics(x) - com
            normal = distance - jnp.dot(distance, cylinder_axis) * cylinder_axis
            return normal / jnp.linalg.norm(normal)

    return cylinder_normal


def create_cylinder_normals(full_kinematics, com, cylinder_axis, sides):
    assert cylinder_axis.shape == (3,)
    cylinder_axis = cylinder_axis / np.linalg.norm(cylinder_axis)
    lat_idx = np.array([side == "lat" for side in sides])
    normals = jnp.zeros((len(sides), 3))

    for idx, side in enumerate(sides):
        if side == "top":
            normals = normals.at[idx, :].set(cylinder_axis)
        elif side == "bottom":
            normals = normals.at[idx, :].set(-cylinder_axis)

    @jit
    def cylinder_normals(x):
        distance = full_kinematics(x)[lat_idx, :] - com
        rejection = distance - jnp.sum(distance * cylinder_axis, axis=1) * cylinder_axis
        rejection_norm = rejection / jnp.linalg.norm(rejection, axis=1, keepdims=True)
        _normals = normals.at[lat_idx, :].set(rejection_norm)
        return _normals

    return cylinder_normals
