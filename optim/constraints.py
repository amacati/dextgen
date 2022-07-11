from typing import Callable, Tuple
from itertools import combinations

import numpy as np
import jax.numpy as jnp
from jax import jit, jacfwd, jacrev, grad
from jax.experimental import host_callback as hcb  # noqa: F401

# jacfwd used forward differentiation mode, jacrev uses reverse-mode. Both are equal up to numerical
# precision, but jacfwd is faster for "tall" jacobians, jacrev is faster for "wide" jacobians. See
# https://github.com/google/jax/issues/47


def create_force_constraints(grasp_forces: Callable) -> Callable:

    def force_constraints(result: np.ndarray, x: np.ndarray, grad: np.ndarray):
        if grad.size > 0:
            grad[:] = _sum_of_forces_jax_jac(x)
        result[:] = _sum_of_forces_jax(x)

    @jit
    def _sum_of_forces_jax(x):
        return jnp.sum(grasp_forces(x)[:, :3], axis=0)

    _sum_of_forces_jax_jac = jit(jacrev(_sum_of_forces_jax))

    return force_constraints


def create_moments_constraints(kinematics: Callable, grasp_forces: Callable,
                               com: np.ndarray) -> Callable:

    def moments_constraints(result: np.ndarray, x: np.ndarray, grad: np.ndarray):
        if grad.size > 0:
            grad[:] = _sum_of_moments_jax_jac(x)
        result[:] = _sum_of_moments_jax(x)

    @jit
    def _sum_of_moments_jax(x):
        cps = kinematics(x)
        wrench = grasp_forces(x)
        f, tau = wrench[:, :3], wrench[:, 3:]
        return jnp.sum(jnp.cross(cps - com, f) + tau, axis=0)

    _sum_of_moments_jax_jac = jit(jacrev(_sum_of_moments_jax))

    return moments_constraints


def create_angle_constraint(grasp_force, cp_normal, max_angle):

    def angle_dot_constraint(x, grad):
        if grad.size > 0:
            grad[:] = _force_angle_dot_jax_grad(x)
        return -max_angle + np.asarray(_force_angle_jax(x)).item()

    @jit
    def _force_angle_jax(x):
        fc = grasp_force(x)[:3]
        normf = fc / jnp.linalg.norm(fc)
        return jnp.arccos(jnp.dot(cp_normal(x), normf))

    _force_angle_dot_jax_grad = jit(grad(_force_angle_jax))

    return angle_dot_constraint


def create_max_angle_constraints(grasp_forces, cp_normals, max_angle):

    def max_angle_constraints(result: np.ndarray, x: np.ndarray, grad: np.ndarray):
        if grad.size > 0:
            grad[:] = _force_angle_jax_jac(x)
        result[:] = _force_angle_jax(x)

    @jit
    def _force_angle_jax(x):
        f = grasp_forces(x)[:, :3]
        f_norm = f / jnp.linalg.norm(f)
        normals = cp_normals(x)
        return jnp.arccos(jnp.sum(f_norm * -normals, axis=1)) - max_angle

    _force_angle_jax_jac = jit(jacrev(_force_angle_jax))

    return max_angle_constraints


def create_plane_constraints(kinematics, plane):
    assert plane.shape == (5, 2, 3)
    offsets = plane[:, 0]
    v_mat = plane[:, 1]

    # TODO: Ask Jan about this
    # for i in range(5):
    #     normal = plane[i, 1]
    #     a_matrix = np.linalg.svd(np.cross(normal, np.identity(normal.shape[0]) * -1))[2]
    #     a_matrix[2, :] = normal / np.linalg.norm(normal)
    #     a_inv = np.linalg.inv(a_matrix.T)
    #     v_mat[i, :] = a_inv[2, :]

    def plane_equality_constraint(x, grad):
        if grad.size > 0:
            grad[:] = _plane_equality_constraint_jax_grad(x)
        return np.asarray(_plane_equality_constraint_jax(x)).item()

    def plane_inequality_constraints(result, x, grad):
        if grad.size > 0:
            grad[:] = _plane_inequality_constraints_jax_jac(x)
        result[:] = _plane_inequality_constraints_jax(x)

    @jit
    def _plane_equality_constraint_jax(x):
        return jnp.dot(kinematics(x) - offsets[0, :], v_mat[0, :])

    _plane_equality_constraint_jax_grad = jit(grad(_plane_equality_constraint_jax))

    @jit
    def _plane_inequality_constraints_jax(x):
        return jnp.sum((kinematics(x) - offsets[1:, :]) * v_mat[1:, :], axis=1)

    _plane_inequality_constraints_jax_jac = jit(jacrev(_plane_inequality_constraints_jax))

    return plane_equality_constraint, plane_inequality_constraints


def create_sphere_constraint(kinematics, offset, radius):

    def sphere_constraint(x, grad):
        if grad.size > 0:
            grad[:] = _radius_constraint_jax_grad(x)
        return np.asarray(_radius_constraint_jax(x)).item()

    @jit
    def _radius_constraint_jax(x):
        return jnp.linalg.norm(kinematics(x) - offset) - radius

    _radius_constraint_jax_grad = jit(grad(_radius_constraint_jax))

    return sphere_constraint


def create_disk_constraints(kinematics, offset, normal, radius) -> Tuple[Callable]:
    assert offset.shape == (3,)
    assert normal.shape == (3,)
    normal = normal / np.linalg.norm(normal)

    def plane_equality_constraint(x, grad):
        if grad.size > 0:
            grad[:] = _plane_equality_constraint_jax_grad(x)
        return np.asarray(_plane_equality_constraint_jax(x)).item()

    @jit
    def _plane_equality_constraint_jax(x):
        return jnp.dot(normal, kinematics(x) - offset)

    _plane_equality_constraint_jax_grad = jit(grad(_plane_equality_constraint_jax))

    sphere_constraint = create_sphere_constraint(kinematics, offset, radius)

    return plane_equality_constraint, sphere_constraint


def create_lateral_surface_constraints(kinematics, cylinder_axis, offsets, radius):
    assert cylinder_axis.shape == (3,)
    normals = np.vstack((cylinder_axis, cylinder_axis, -cylinder_axis))
    assert offsets.shape == (3, 3)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    def distance_constraint(x, grad):
        if grad.size > 0:
            grad[:] = _lateral_distance_constraint_jax_grad(x)
        return np.asarray(_lateral_distance_constraint_jax(x)).item()

    def _lateral_distance_constraint_jax(x):
        return jnp.linalg.norm(jnp.cross(kinematics(x) - offsets[0, :], normals[0, :])) - radius

    _lateral_distance_constraint_jax_grad = jit(grad(_lateral_distance_constraint_jax))

    def plane_inequality_constraints(result, x, grad):
        if grad.size > 0:
            grad[:] = _plane_inequality_constraints_jax_jac(x)
        result[:] = _plane_inequality_constraints_jax(x)

    @jit
    def _plane_inequality_constraints_jax(x):
        return jnp.sum((kinematics(x) - offsets[1:, ...]) * normals[1:, ...], axis=1)

    _plane_inequality_constraints_jax_jac = jit(jacfwd(_plane_inequality_constraints_jax))

    return distance_constraint, plane_inequality_constraints


def create_distance_constraints(full_kinematics: Callable, min_dist: float) -> Callable:

    def distance_constraints(result: np.ndarray, x: np.ndarray, grad: np.ndarray):
        if grad.size > 0:
            grad[:] = _distance_jax_jac(x)
        result[:] = _distance_jax(x)

    @jit
    def _distance_jax(x):
        cps = full_kinematics(x)
        return min_dist - jnp.array([jnp.linalg.norm(p[0] - p[1]) for p in combinations(cps, 2)])

    _distance_jax_jac = jit(jacrev(_distance_jax))

    return distance_constraints
