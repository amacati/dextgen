"""Constraints collection module."""
from typing import Callable, Tuple
from itertools import combinations

import numpy as np
import jax.numpy as jnp


def create_force_constraints(grasp_forces: Callable) -> Callable:
    """Create force constraints restricting the sum of forces to be zero.

    Args:
        grasp_forces: Callable calculating all grasp forces from the gripper config.

    Returns:
        The force equality constraint callable.
    """

    def force_constraints(x: jnp.ndarray) -> jnp.ndarray:
        """Calculate the force constraints.

        Args:
            x: Optimization variable.

        Returns:
            The sum of all forces.
        """
        return jnp.sum(grasp_forces(x)[:, :3], axis=0)

    return force_constraints


def create_moments_constraints(kinematics: Callable, grasp_forces: Callable,
                               com: np.ndarray) -> Callable:
    """Create moment constraints restricting the sum of moments to be zero.

    Args:
        kinematics: Callable calculating all contact point positions from the gripper config.
        grasp_forces: Callable calculating all grasp forces from the gripper config.
        com: Object center of mass

    Returns:
        The moments equality constraint callable.
    """

    def moments_constraints(x: jnp.ndarray) -> jnp.ndarray:
        """Calculate the moments constraints.

        Args:
            x: Optimization variable.

        Returns:
            The sum of all moments.
        """
        con_pts = kinematics(x)
        wrench = grasp_forces(x)
        f, tau = wrench[:, :3], wrench[:, 3:]
        return jnp.sum(jnp.cross(con_pts - com, f) + tau, axis=0)

    return moments_constraints


def create_max_angle_constraints(grasp_forces: Callable, cp_normals: Callable,
                                 max_angle: float) -> Callable:
    """Create angle constraints restricting the contact force angle to be smaller than a maximum.

    Args:
        grasp_forces: Callable calculating all grasp forces from the gripper config.
        cp_normals: Callable calculating the contact surface normal at the current contact point.
        max_angle: The maximum allowed angle.

    Returns:
        The angle inequality constraint callable.
    """
    max_angle_c = jnp.cos(max_angle)

    def max_angle_constraint(x: jnp.ndarray) -> jnp.ndarray:
        """Calculate the maximum angle constraints.

        Args:
            x: Optimization variable.

        Returns:
            The difference from the angles' cosine to the maximum angles' cosine.
        """
        f = grasp_forces(x)[:, :3]
        normals = cp_normals(x)
        angle_c = jnp.sum(f * -normals, axis=1) / jnp.linalg.norm(f, axis=1)
        return angle_c - max_angle_c

    return max_angle_constraint


def create_plane_constraints(kinematics: Callable, offsets: np.ndarray,
                             normals: np.ndarray) -> Callable:
    """Create a plane constraint restricting the contact point to a finite, rectangular plane.

    The constraint is constructed by constraining the point to a plane with an equality constraint,
    and four border inequality constraints on each side of the rectangle.

    Args:
        kinematics: Callable calculating all contact point positions from the gripper config.
        offsets: Plane offsets for the plane itself and its four restricting border planes.
        normals: Plane normals for the plane itself and its four restricting border planes.

    Returns:
        The plane equality constraint callable and the border inequality constraints callable.
    """
    assert offsets.shape == (5, 3)
    assert normals.shape == (5, 3)

    def plane_equality_constraint(x: jnp.ndarray) -> float:
        """Calculate the plane constraint.

        Args:
            x: Optimization variable.

        Returns:
            The distance of the contact point to the plane.
        """
        return jnp.dot(kinematics(x) - offsets[0], normals[0])

    def plane_inequality_constraints(x: jnp.ndarray) -> jnp.ndarray:
        """Calculate the plane inequality constraints.

        Args:
            x: Optimization variable.

        Returns:
            The vector of distances to each boundary plane.
        """
        # Negative since contact points are expected to be "behind" the normal of the boundary plane
        return -jnp.sum((kinematics(x) - offsets[1:]) * normals[1:], axis=1)

    return plane_equality_constraint, plane_inequality_constraints


def create_sphere_constraint(kinematics: Callable, offset: np.ndarray, radius: float) -> Callable:
    """Create a sphere constraint restricting the contact point to a sphere.

    Args:
        kinematics: Callable calculating all contact point positions from the gripper config.
        offset: Sphere center offset.
        radius: Sphere radius.

    Returns:
        The sphere equality constraint callable.
    """

    def sphere_constraint(x: jnp.ndarray) -> jnp.ndarray:
        """Calculate the sphere constraint.

        Args:
            x: Optimization variable.

        Returns:
            The distance of the contact point to the surface of the sphere.
        """
        return jnp.linalg.norm(kinematics(x) - offset) - radius

    return sphere_constraint


def create_disk_constraints(kinematics: Callable, offset: np.ndarray, normal: np.ndarray,
                            radius: float) -> Tuple[Callable]:
    """Create a disk constraint restricting the contact point to a disk surface.

    Consists of two constraints constricting the contact to lie on a plane and within a sphere.

    Args:
        kinematics: Callable calculating all contact point positions from the gripper config.
        offset: Disk center offset.
        normal: Disk plane normal.
        radius: Disk radius.

    Returns:
        The disk equality constraint callable and the disk inequality constraint callable.
    """
    assert offset.shape == (3,)
    assert normal.shape == (3,)
    normal = normal / np.linalg.norm(normal)

    def plane_equality_constraint(x: jnp.ndarray) -> float:
        """Calculate the plane constraint.

        Args:
            x: Optimization variable.

        Returns:
            The distance of the contact point to the plane.
        """
        return jnp.dot(normal, kinematics(x) - offset)

    sphere_constraint = create_sphere_constraint(kinematics, offset, radius)

    return plane_equality_constraint, sphere_constraint


def create_distance_constraints(full_kinematics: Callable, min_dist: float) -> Callable:
    """Create distance constraints between the contact points.

    Args:
        full_kinematics: Callable to calculate all contact points from the optimization config.
        min_dist: Minimal distance between contact points.

    Returns:
        The constraint callable.
    """

    def distance_constraints(x: jnp.ndarray) -> jnp.ndarray:
        """Calculate the contact points constraints.

        Takes the norm of each contact point pair distance and substracts the minimum distance.

        Args:
            x: Optimization variable.

        Returns:
            A vector of all distance constraint values.
        """
        cps = full_kinematics(x)
        return jnp.array([jnp.linalg.norm(p[0] - p[1]) for p in combinations(cps, 2)]) - min_dist

    return distance_constraints


def quaternion_cnst(x: jnp.ndarray) -> float:
    """Calculate the unit quaternion constraint.

    Args:
        x: Optimization variable. Assumes the quaternion at indices [3:7].

    Returns:
        The norm deviation from 1.
    """
    return jnp.sum(x[3:7]**2) - 1
