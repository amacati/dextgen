"""Parallel Jaw gripper kinematics module."""
from typing import Tuple, Union

import numpy as np
import jax.numpy as jnp

from optim.grippers.kinematics.tf import tf_matrix, tf_matrix_q

PJ_JOINT_LIMITS = {"lower": np.array([0, 0]), "upper": np.array([0.05, 0.05])}
JOINT_RANGE = 0.05

# contact 0.015 offset because 0.008 offset + 0.007 box thickness
ROOT_T_FR = tf_matrix(np.array([0., 0.0159, 0.1, 0., np.pi / 2, 0]))
ROOT_T_FL = tf_matrix(np.array([0., -0.0159, 0.1, 0., np.pi / 2, 0]))
FR_T_FRCON = tf_matrix(np.array([-0.01925, -0.015, 0., 0., 0., 0.]))
FL_T_FLCON = tf_matrix(np.array([-0.01925, 0.015, 0., 0., 0., 0.]))

T = Union[np.ndarray, jnp.ndarray]


def kin_pj_full(x: T) -> Union[Tuple[np.ndarray], Tuple[jnp.ndarray]]:
    """Compute all link frames of the ParallelJaw gripper.

    Args:
        x: Joint configuration.

    Returns:
        The joint frames of all links.
    """
    w_T_root = tf_matrix_q(x[:7])
    w_T_fr = w_T_root @ ROOT_T_FR @ tf_matrix(jnp.array([0, x[7], 0, 0, 0, 0]))
    w_T_frcon = w_T_fr @ FR_T_FRCON
    w_T_fl = w_T_root @ ROOT_T_FL @ tf_matrix(jnp.array([0, -x[8], 0, 0, 0, 0]))
    w_T_flcon = w_T_fl @ FL_T_FLCON
    return w_T_root, w_T_fr, w_T_frcon, w_T_fl, w_T_flcon


def kin_pj_right(x: T) -> T:
    """Compute the frame of the ParallelJaw gripper's right finger.

    Frame position is located at the center of the inner surface of the finger.

    Args:
        x: Joint configuration.

    Returns:
        The finger frame.
    """
    w_T_root = tf_matrix_q(x[:7])
    w_T_fr = w_T_root @ ROOT_T_FR @ tf_matrix(jnp.array([0, x[7], 0, 0, 0, 0]))
    w_T_frcon = w_T_fr @ FR_T_FRCON
    return w_T_frcon


def kin_pj_left(x: T) -> T:
    """Compute the frame of the ParallelJaw gripper's left finger.

    Frame position is located at the center of the inner surface of the finger.

    Args:
        x: Joint configuration.

    Returns:
        The finger frame.
    """
    w_T_root = tf_matrix_q(x[:7])
    w_T_fl = w_T_root @ ROOT_T_FL @ tf_matrix(jnp.array([0, -x[8], 0, 0, 0, 0]))
    w_T_flcon = w_T_fl @ FL_T_FLCON
    return w_T_flcon
