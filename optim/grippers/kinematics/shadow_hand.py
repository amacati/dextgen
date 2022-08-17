"""ShadowHand kinematics module."""
from typing import Tuple, Union

import numpy as np
from jax import jit
import jax.numpy as jnp

from optim.grippers.kinematics.tf import tf_matrix, tf_matrix_q, zrot_matrix

JOINT_LIMITS = {
    "lower":
        np.array([
            -0.489, -0.785, -0.960, 0, -0.209, -0.436, 0, -0.349, 0, 0, -0.349, 0, 0, -0.349, 0, 0,
            0, -0.349, 0, 0
        ]),
    "upper":
        np.array([
            0.140, 0.524, 0.960, 1.222, 0.209, 0.436, 1.571, 0.349, 1.571, 1.571, 0.349, 1.571,
            1.571, 0.349, 1.571, 1.571, 0.785, 0.349, 1.571, 1.571
        ])
}

COUPLING_CONST = 0.87577639751

WRIST2_T_WRIST1 = tf_matrix(np.array([0.034, 0, 0, -np.pi / 2, 0, 0]))
# Thumb
WRIST1_T_TH5 = tf_matrix(np.array([0.029, 0.0085, 0.034, 0, 45, 0]))
TH5_T_TH4 = tf_matrix(np.array([0, 0, 0, 0, -np.pi / 2, 0]))
TH4_T_TH3 = tf_matrix(np.array([0.038, 0, 0, 0, 0, 0]))
TH3_T_TH2 = tf_matrix(np.array([0, 0, 0, np.pi / 2, 0, 0]))
TH2_T_TH1 = tf_matrix(np.array([0.032, 0, 0, 0, 0, 0]))
TH1_T_TH0 = tf_matrix(np.array([0.0275, 0, 0, 0, 0, 0]))
# Index finger
WRIST1_T_FF4 = tf_matrix(np.array([0.095, 0, 0.034, np.pi / 2, 0, 0]))
FF4_T_FF3 = tf_matrix(np.array([0, 0, 0, -np.pi / 2, 0, 0]))
FF3_T_FF2 = tf_matrix(np.array([0.045, 0, 0, 0, 0, 0]))
FF2_T_FF1 = tf_matrix(np.array([0.025, 0, 0, 0, 0, 0]))
FF1_T_FF0 = tf_matrix(np.array([0.026, 0, 0, 0, 0, 0]))
# Middle finger
WRIST1_T_MF4 = tf_matrix(np.array([0.099, 0, 0.012, np.pi / 2, 0, 0]))
MF4_T_MF3 = tf_matrix(np.array([0, 0, 0, -np.pi / 2, 0, 0]))
MF3_T_MF2 = tf_matrix(np.array([0.045, 0, 0, 0, 0, 0]))
MF2_T_MF1 = tf_matrix(np.array([0.025, 0, 0, 0, 0, 0]))
MF1_T_MF0 = tf_matrix(np.array([0.026, 0, 0, 0, 0, 0]))
# Ring finger
WRIST1_T_RF4 = tf_matrix(np.array([0.095, 0, -0.01, np.pi / 2, 0, 0]))
RF4_T_RF3 = tf_matrix(np.array([0, 0, 0, -np.pi / 2, 0, 0]))
RF3_T_RF2 = tf_matrix(np.array([0.045, 0, 0, 0, 0, 0]))
RF2_T_RF1 = tf_matrix(np.array([0.025, 0, 0, 0, 0, 0]))
RF1_T_RF0 = tf_matrix(np.array([0.026, 0, 0, 0, 0, 0]))
# Little finger
WRIST1_T_LF5 = tf_matrix(np.array([0.02071, 0, -0.032, 0, 11 * np.pi / 36, 0]))
LF5_T_LF4 = tf_matrix(np.array([0.037792, 0, 0.053974, np.pi / 2, -11 * np.pi / 36, 0]))
LF4_T_LF3 = tf_matrix(np.array([0, 0, 0, -np.pi / 2, 0, 0]))
LF3_T_LF2 = tf_matrix(np.array([0.045, 0, 0, 0, 0, 0]))
LF2_T_LF1 = tf_matrix(np.array([0.025, 0, 0, 0, 0, 0]))
LF1_T_LF0 = tf_matrix(np.array([0.026, 0, 0, 0, 0, 0]))


@jit
def kin_sh_full(x: Union[np.ndarray, jnp.ndarray]) -> Union[Tuple[np.ndarray], Tuple[jnp.ndarray]]:
    """Compute all link frames of the ShadowHand.

    See https://www.shadowrobot.com/wp-content/uploads/2022/03/shadow_dexterous_hand_e_technical_specification.pdf  # noqa: E501
    for frame names and numbers. Finger tips are denoted with 0.

    Args:
        x: Joint configuration.

    Returns:
        The joint frames of all links.
    """
    w_T_root = tf_matrix_q(x[:7])
    w_T_wrist2 = w_T_root @ zrot_matrix(x[7])
    w_T_wrist1 = w_T_wrist2 @ WRIST2_T_WRIST1 @ zrot_matrix(x[8])
    # Thumb
    w_T_th5 = w_T_wrist1 @ WRIST1_T_TH5 @ zrot_matrix(x[9])
    w_T_th4 = w_T_th5 @ TH5_T_TH4 @ zrot_matrix(x[10])
    w_T_th3 = w_T_th4 @ TH4_T_TH3 @ zrot_matrix(x[11])
    w_T_th2 = w_T_th3 @ TH3_T_TH2 @ zrot_matrix(x[12])
    w_T_th1 = w_T_th2 @ TH2_T_TH1 @ zrot_matrix(x[13])
    w_T_th0 = w_T_th1 @ TH1_T_TH0  # Unactuated
    # Index finger
    w_T_ff4 = w_T_wrist1 @ WRIST1_T_FF4 @ zrot_matrix(x[14])
    w_T_ff3 = w_T_ff4 @ FF4_T_FF3 @ zrot_matrix(x[15])
    w_T_ff2 = w_T_ff3 @ FF3_T_FF2 @ zrot_matrix(x[16])
    w_T_ff1 = w_T_ff2 @ FF2_T_FF1 @ zrot_matrix(x[16] * COUPLING_CONST)
    w_T_ff0 = w_T_ff1 @ FF1_T_FF0  # Unactuated
    # Middle finger
    w_T_mf4 = w_T_wrist1 @ WRIST1_T_MF4 @ zrot_matrix(x[17])
    w_T_mf3 = w_T_mf4 @ MF4_T_MF3 @ zrot_matrix(x[18])
    w_T_mf2 = w_T_mf3 @ MF3_T_MF2 @ zrot_matrix(x[19])
    w_T_mf1 = w_T_mf2 @ MF2_T_MF1 @ zrot_matrix(x[19] * COUPLING_CONST)
    w_T_mf0 = w_T_mf1 @ MF1_T_MF0  # Unactuated
    # Ring finger
    w_T_rf4 = w_T_wrist1 @ WRIST1_T_RF4 @ zrot_matrix(x[20])
    w_T_rf3 = w_T_rf4 @ RF4_T_RF3 @ zrot_matrix(x[21])
    w_T_rf2 = w_T_rf3 @ RF3_T_RF2 @ zrot_matrix(x[22])
    w_T_rf1 = w_T_rf2 @ RF2_T_RF1 @ zrot_matrix(x[22] * COUPLING_CONST)
    w_T_rf0 = w_T_rf1 @ RF1_T_RF0  # Unactuated
    # Little finger
    w_T_lf5 = w_T_wrist1 @ WRIST1_T_LF5 @ zrot_matrix(x[23])
    w_T_lf4 = w_T_lf5 @ LF5_T_LF4 @ zrot_matrix(x[24])
    w_T_lf3 = w_T_lf4 @ LF4_T_LF3 @ zrot_matrix(x[25])
    w_T_lf2 = w_T_lf3 @ LF3_T_LF2 @ zrot_matrix(x[26])
    w_T_lf1 = w_T_lf2 @ LF2_T_LF1 @ zrot_matrix(x[26] * COUPLING_CONST)
    w_T_lf0 = w_T_lf1 @ LF1_T_LF0  # Unactuated
    return (w_T_root, w_T_wrist2, w_T_wrist1, w_T_th5, w_T_th4, w_T_th3, w_T_th2, w_T_th1, w_T_th0,
            w_T_ff4, w_T_ff3, w_T_ff2, w_T_ff1, w_T_ff0, w_T_mf4, w_T_mf3, w_T_mf2, w_T_mf1,
            w_T_mf0, w_T_rf4, w_T_rf3, w_T_rf2, w_T_rf1, w_T_rf0, w_T_lf5, w_T_lf4, w_T_lf3,
            w_T_lf2, w_T_lf1, w_T_lf0)
