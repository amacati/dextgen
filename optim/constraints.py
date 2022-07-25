from typing import Callable, Tuple
from itertools import combinations

import numpy as np
import jax.numpy as jnp


def create_force_constraints(grasp_forces: Callable) -> Callable:

    def force_constraints(x):
        return jnp.sum(grasp_forces(x)[:, :3], axis=0)

    return force_constraints


def create_moments_constraints(kinematics: Callable, grasp_forces: Callable,
                               com: np.ndarray) -> Callable:

    def moments_constraints(x):
        cps = kinematics(x)
        wrench = grasp_forces(x)
        f, tau = wrench[:, :3], wrench[:, 3:]
        return jnp.sum(jnp.cross(cps - com, f) + tau, axis=0)

    return moments_constraints


def create_max_angle_constraints(grasp_forces, cp_normals, max_angle):
    max_angle_c = jnp.cos(max_angle)

    def max_angle_constraint(x):
        f = grasp_forces(x)[:, :3]
        normals = cp_normals(x)
        angle_c = jnp.sum(f * -normals, axis=1) / jnp.linalg.norm(f, axis=1)
        return angle_c - max_angle_c

    return max_angle_constraint


def create_plane_constraints(kinematics, offsets, normals):
    assert offsets.shape == (5, 3)
    assert normals.shape == (5, 3)

    def plane_equality_constraint(x):
        return jnp.dot(kinematics(x) - offsets[0], normals[0])

    def plane_inequality_constraints(x):
        # Negative since contact points are expected to be "behind" the normal of the boundary plane
        return -jnp.sum((kinematics(x) - offsets[1:]) * normals[1:], axis=1)

    return plane_equality_constraint, plane_inequality_constraints


def create_sphere_constraint(kinematics, offset, radius):

    def sphere_constraint(x):
        return jnp.linalg.norm(kinematics(x) - offset) - radius

    return sphere_constraint


def create_disk_constraints(kinematics, offset, normal, radius) -> Tuple[Callable]:
    assert offset.shape == (3,)
    assert normal.shape == (3,)
    normal = normal / np.linalg.norm(normal)

    def plane_equality_constraint(x):
        return jnp.dot(normal, kinematics(x) - offset)

    sphere_constraint = create_sphere_constraint(kinematics, offset, radius)

    return plane_equality_constraint, sphere_constraint


def create_lateral_surface_constraints(kinematics, cylinder_axis, offsets, radius):
    assert cylinder_axis.shape == (3,)
    normals = np.vstack((cylinder_axis, cylinder_axis, -cylinder_axis))
    assert offsets.shape == (3, 3)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    def lateral_surface_constraint(x):
        return jnp.linalg.norm(jnp.cross(kinematics(x) - offsets[0, :], normals[0, :])) - radius

    def plane_inequality_constraints(x):
        return -jnp.sum((kinematics(x) - offsets[1:, :]) * normals[1:, :], axis=1)

    return lateral_surface_constraint, plane_inequality_constraints


def create_distance_constraints(full_kinematics: Callable, min_dist: float) -> Callable:

    def distance_constraints(x):
        cps = full_kinematics(x)
        return jnp.array([jnp.linalg.norm(p[0] - p[1]) for p in combinations(cps, 2)]) - min_dist

    return distance_constraints


def quaternion_cnst(x):
    return jnp.sum(x[3:7]**2) - 1
