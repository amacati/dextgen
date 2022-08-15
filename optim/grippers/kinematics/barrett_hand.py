"""BarrettHand kinematics module."""
from typing import Tuple, Union

import numpy as np
import jax.numpy as jnp

from optim.grippers.kinematics.tf import tf_matrix, tf_matrix_q, zrot_matrix

BH_JOINT_LIMITS = {
    "lower": np.array([0., 0., 0., 0.]),
    "upper": np.array([np.pi, 2.44346, 2.44346, 2.44346])
}

COUPLING_CONST = 1 / 3

PALM_T_F1PROX = tf_matrix(np.array([-0.025, 0, 0.0415, 0, 0, -np.pi / 2]))
PALM_T_F2PROX = tf_matrix(np.array([0.025, 0, 0.0415, 0, 0, -np.pi / 2]))
PALM_T_F3MED = tf_matrix(np.array([0, 0.05, 0.0754, 0, np.pi / 2, 0])) @ zrot_matrix(np.pi / 2)
FPROX_T_FMED = tf_matrix(np.array([0.05, 0, 0.0339, np.pi / 2, 0, 0]))
FMED_T_FDIST = tf_matrix(np.array([0.06994, 0.003, 0, 0, 0, 0]))
FDIST_T_LAST = tf_matrix(np.array([0.059, 0, 0, 0, 0, 0]))

T = Union[np.ndarray, jnp.ndarray]


def kin_1dist_p(x: T) -> T:
    """Compute the frame of the BarrettHand's first finger tip.

    Args:
        x: Joint configuration.

    Returns:
        The finger tip frame.
    """
    w_T_root = tf_matrix_q(x[0:7])
    w_T_f1prox = w_T_root @ PALM_T_F1PROX @ zrot_matrix(-x[7])
    w_T_f1med = w_T_f1prox @ FPROX_T_FMED @ zrot_matrix(x[8])
    w_T_f1dist = w_T_f1med @ FMED_T_FDIST @ zrot_matrix(x[8] * COUPLING_CONST)
    w_T_f1last = w_T_f1dist @ FDIST_T_LAST
    return w_T_f1last


def kin_2dist_p(x: T) -> T:
    """Compute the frame of the BarrettHand's second finger tip.

    Args:
        x: Joint configuration.

    Returns:
        The finger tip frame.
    """
    w_T_root = tf_matrix_q(x[0:7])
    w_T_f2prox = w_T_root @ PALM_T_F2PROX @ zrot_matrix(x[7])
    w_T_f2med = w_T_f2prox @ FPROX_T_FMED @ zrot_matrix(x[9])
    w_T_f2dist = w_T_f2med @ FMED_T_FDIST @ zrot_matrix(x[9] * COUPLING_CONST)
    w_T_f2last = w_T_f2dist @ FDIST_T_LAST
    return w_T_f2last


def kin_3dist_p(x: T) -> T:
    """Compute the frame of the BarrettHand's third finger tip.

    Args:
        x: Joint configuration.

    Returns:
        The finger tip frame.
    """
    w_T_root = tf_matrix_q(x[0:7])
    w_T_f3med = w_T_root @ PALM_T_F3MED @ zrot_matrix(x[10])
    w_T_f3dist = w_T_f3med @ FMED_T_FDIST @ zrot_matrix(x[10] * COUPLING_CONST)
    w_T_f3last = w_T_f3dist @ FDIST_T_LAST
    return w_T_f3last


def kin_bh_full(x: T) -> Union[Tuple[np.ndarray], Tuple[jnp.ndarray]]:
    """Compute all link frames of the BarrettHand.

    Args:
        x: Joint configuration.

    Returns:
        The joint frames of all links.
    """
    w_T_root = tf_matrix_q(x[0:7])
    # Finger 1
    w_T_f1prox = w_T_root @ PALM_T_F1PROX @ zrot_matrix(-x[7])
    w_T_f1med = w_T_f1prox @ FPROX_T_FMED @ zrot_matrix(x[8])
    w_T_f1dist = w_T_f1med @ FMED_T_FDIST @ zrot_matrix(x[8] * COUPLING_CONST)
    w_T_f1last = w_T_f1dist @ FDIST_T_LAST

    # Finger 2
    w_T_f2prox = w_T_root @ PALM_T_F2PROX @ zrot_matrix(x[7])
    w_T_f2med = w_T_f2prox @ FPROX_T_FMED @ zrot_matrix(x[9])
    w_T_f2dist = w_T_f2med @ FMED_T_FDIST @ zrot_matrix(x[9] * COUPLING_CONST)
    w_T_f2last = w_T_f2dist @ FDIST_T_LAST

    # Finger 3
    w_T_f3med = w_T_root @ PALM_T_F3MED @ zrot_matrix(x[10])
    w_T_f3dist = w_T_f3med @ FMED_T_FDIST @ zrot_matrix(x[10] * COUPLING_CONST)
    w_T_f3last = w_T_f3dist @ FDIST_T_LAST

    return (w_T_root, w_T_f1prox, w_T_f1med, w_T_f1dist, w_T_f1last, w_T_f2prox, w_T_f2med,
            w_T_f2dist, w_T_f2last, w_T_f3med, w_T_f3dist, w_T_f3last)
