import numpy as np
from jax import jit
from optim.kinematics.tf import tf_matrix

PJ_JOINT_LIMITS = {"lower": np.array([0]), "upper": np.array([0.05])}
JOINT_RANGE = 0.05

F_T_FLAST = tf_matrix(0.08, 0, 0, 0, 0, 0)


@jit
def pj_kinematics(x):
    w_T_root = tf_matrix(*x[0:6])

    # Finger 1
    w_T_f1 = w_T_root @ tf_matrix(0, 0, x[6] * JOINT_RANGE, 0, 0, 0)
    w_T_f1last = w_T_f1 @ F_T_FLAST

    # Finger 2
    w_T_f2 = w_T_root @ tf_matrix(0, 0, -x[6] * JOINT_RANGE, 0, 0, 0)
    w_T_f2last = w_T_f2 @ F_T_FLAST
    return w_T_root, w_T_f1, w_T_f1last, w_T_f2, w_T_f2last
