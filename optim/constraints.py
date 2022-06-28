from typing import Callable, Tuple
from itertools import combinations

import numpy as np
import jax.numpy as jnp
from jax import jit, jacfwd, grad


def create_force_constraints(grasp_forces: Callable) -> Callable:

    @jit
    def _sum_of_forces_jax(x):
        f = grasp_forces(x)
        return jnp.sum(f, axis=0)

    _sum_of_forces_jax_jac = jit(jacfwd(_sum_of_forces_jax))

    def force_constraints(result: np.ndarray, x: np.ndarray, grad: np.ndarray):
        if grad.size > 0:
            grad[:] = _sum_of_forces_jax_jac(x)
        result[:] = _sum_of_forces_jax(x)

    return force_constraints


def create_moments_constraints(com: np.ndarray, grasp_forces: Callable) -> Callable:

    @jit
    def _sum_of_moments_jax(x):
        f = grasp_forces(x)
        return jnp.sum(jnp.cross(com - f, f), axis=1)

    _sum_of_moments_jax_jac = jit(jacfwd(_sum_of_moments_jax))

    def moments_constraints(result: np.ndarray, x: np.ndarray, grad: np.ndarray):
        if grad.size > 0:
            grad[:] = _sum_of_moments_jax_jac(x)
        result[:] = _sum_of_moments_jax(x)

    return moments_constraints


def create_maximum_force_constraints(fmax: float) -> Callable:

    def maximum_force_constraints(result: np.ndarray, x: np.ndarray, grad: np.ndarray):
        if grad.size > 0:
            grad[:] = _force_norm_jax_jac(x)
        result[:] = _force_norm_jax(x) - fmax

    return maximum_force_constraints


def create_minimum_force_constraints(fmin: float) -> Callable:

    def minimum_force_constraints(result: np.ndarray, x: np.ndarray, grad: np.ndarray):
        if grad.size > 0:
            grad[:] = -_force_norm_jax_jac(x)
        result[:] = fmin - _force_norm_jax(x)

    return minimum_force_constraints


@jit
def _force_norm_jax(x):
    return jnp.linalg.norm(x.reshape(-1, 6)[:, 3:], axis=1)


_force_norm_jax_jac = jit(jacfwd(_force_norm_jax))


def homogeneous_forces_contraint(x: np.ndarray, grad: np.ndarray):
    if grad.size > 0:
        grad[:] = _homogeneous_force_jax_grad(x)
    return np.asarray(_homogeneous_force_jax(x)).item()


@jit
def _homogeneous_force_jax(x):
    fnorm = _force_norm_jax(x)
    return jnp.sum((fnorm - jnp.mean(fnorm))**2)


_homogeneous_force_jax_grad = jit(grad(_homogeneous_force_jax))


def create_angle_constraint(grasp_force, cp_normal, max_angle_cos):

    def angle_dot_constraint(x, grad):
        if grad.size > 0:
            grad[:] = _force_angle_dot_jax_grad(x)
        return max_angle_cos + np.asarray(_force_angle_dot_jax(x)).item()

    @jit
    def _force_angle_dot_jax(x):
        fc = grasp_force(x)
        normf = fc / jnp.linalg.norm(fc)
        return jnp.dot(cp_normal(x), normf)

    _force_angle_dot_jax_grad = jit(grad(_force_angle_dot_jax))

    return angle_dot_constraint


def create_plane_constraints(kinematics, plane):
    assert plane.shape == (5, 2, 3)
    offsets = plane[:, 0]
    v_mat = np.zeros((5, 3))
    for i in range(5):
        normal = plane[i, 1]
        a_matrix = np.linalg.svd(np.cross(normal, np.identity(normal.shape[0]) * -1))[2]
        a_matrix[2, :] = normal / np.linalg.norm(normal)
        a_inv = np.linalg.inv(a_matrix.T)
        v_mat[i, :] = a_inv[2, :]

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
        cp = kinematics(x)
        return jnp.dot(v_mat[0, :], cp - offsets[0, :])

    _plane_equality_constraint_jax_grad = jit(grad(_plane_equality_constraint_jax))

    @jit
    def _plane_inequality_constraints_jax(x):
        cp = kinematics(x)
        return jnp.sum((cp - offsets[1:, ...]) * v_mat[1:, ...], axis=1)

    _plane_inequality_constraints_jax_jac = jit(jacfwd(_plane_inequality_constraints_jax))

    return plane_equality_constraint, plane_inequality_constraints


def create_sphere_constraint(cp_idx, offset, radius):

    def sphere_constraint(x, grad):
        if grad.size > 0:
            grad[:] = _radius_constraint_jax_grad(x)
        return np.asarray(_radius_constraint_jax(x)).item()

    @jit
    def _radius_constraint_jax(x):
        cp = x[cp_idx * 6:cp_idx * 6 + 3]
        return jnp.linalg.norm(cp - offset) - radius

    _radius_constraint_jax_grad = jit(grad(_radius_constraint_jax))

    return sphere_constraint


def create_disk_constraints(cp_idx, offset, normal, radius) -> Tuple[Callable]:
    assert offset.shape == (3,)
    assert normal.shape == (3,)
    normal = normal / np.linalg.norm(normal)

    def plane_equality_constraint(x, grad):
        if grad.size > 0:
            grad[:] = _plane_equality_constraint_jax_grad(x)
        return np.asarray(_plane_equality_constraint_jax(x)).item()

    @jit
    def _plane_equality_constraint_jax(x):
        cp = x[cp_idx * 6:cp_idx * 6 + 3]
        return jnp.dot(normal, cp - offset)

    _plane_equality_constraint_jax_grad = jit(grad(_plane_equality_constraint_jax))

    sphere_constraint = create_sphere_constraint(cp_idx, offset, radius)

    return plane_equality_constraint, sphere_constraint


def create_lateral_surface_constraints(cp_idx, cylinder_axis, offsets, radius):
    assert cylinder_axis.shape == (3,)
    normals = np.vstack((cylinder_axis, cylinder_axis, -cylinder_axis))
    assert offsets.shape == (3, 3)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    def distance_constraint(x, grad):
        if grad.size > 0:
            grad[:] = _lateral_distance_constraint_jax_grad(x)
        return np.asarray(_lateral_distance_constraint_jax(x)).item()

    def _lateral_distance_constraint_jax(x):
        cp = x[cp_idx * 6:cp_idx * 6 + 3]
        return jnp.linalg.norm(jnp.cross(cp - offsets[0, :], normals[0, :])) - radius

    _lateral_distance_constraint_jax_grad = jit(grad(_lateral_distance_constraint_jax))

    def plane_inequality_constraints(result, x, grad):
        if grad.size > 0:
            grad[:] = _plane_inequality_constraints_jax_jac(x)
        result[:] = _plane_inequality_constraints_jax(x)

    @jit
    def _plane_inequality_constraints_jax(x):
        cp = x[cp_idx * 6:cp_idx * 6 + 3]
        return jnp.sum((cp - offsets[1:, ...]) * normals[1:, ...], axis=1)

    _plane_inequality_constraints_jax_jac = jit(jacfwd(_plane_inequality_constraints_jax))

    return distance_constraint, plane_inequality_constraints


def create_distance_constraints(min_dist: float) -> Callable:

    def distance_constraints(result: np.ndarray, x: np.ndarray, grad: np.ndarray):
        if grad.size > 0:
            grad[:] = _distance_jax_jac(x)
        result[:] = _distance_jax(x)

    @jit
    def _distance_jax(x):
        pts = x.reshape(-1, 6)[:, :3]
        return min_dist - jnp.array([jnp.linalg.norm(x[0] - x[1]) for x in combinations(pts, 2)])

    _distance_jax_jac = jit(jacfwd(_distance_jax))

    return distance_constraints
