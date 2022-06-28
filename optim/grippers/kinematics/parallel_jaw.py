import numpy as np
from jax import jit
from optim.grippers.kinematics.tf import tf_matrix, tf_matrix_q

PJ_JOINT_LIMITS = {"lower": np.array([0]), "upper": np.array([0.05])}
JOINT_RANGE = 0.05


def pj_kinematics(link):
    return _kinematics_left if link == "robot0:l_gripper_finger_link" else _kinematics_right


@jit
def _kinematics_left(x):
    w_T_root = tf_matrix_q(*x[:7])
    return w_T_root @ tf_matrix(0, 0, -x[7] * JOINT_RANGE, 0, 0, 0)


@jit
def _kinematics_right(x):
    w_T_root = tf_matrix_q(*x[:7])
    return w_T_root @ tf_matrix(0, 0, x[7] * JOINT_RANGE, 0, 0, 0)