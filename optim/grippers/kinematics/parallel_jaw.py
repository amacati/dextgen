import numpy as np
import jax.numpy as jnp
from optim.grippers.kinematics.tf import tf_matrix, tf_matrix_q

PJ_JOINT_LIMITS = {"lower": np.array([0]), "upper": np.array([0.05])}
JOINT_RANGE = 0.05

# contact 0.015 offset because 0.008 offset + 0.007 box thickness
root_T_fr = tf_matrix(np.array([0., 0.0159, 0.1, 0., np.pi / 2, 0]))
root_T_fl = tf_matrix(np.array([0., -0.0159, 0.1, 0., np.pi / 2, 0]))
fr_T_frcon = tf_matrix(np.array([-0.01925, -0.015, 0., 0., 0., 0.]))
fl_T_frcon = tf_matrix(np.array([-0.01925, 0.015, 0., 0., 0., 0.]))


def kin_pj_full(x):
    w_T_root = tf_matrix_q(x[:7])
    w_T_fr = w_T_root @ root_T_fr @ tf_matrix(jnp.array([0, x[7], 0, 0, 0, 0]))
    w_T_frcon = w_T_fr @ fr_T_frcon
    w_T_fl = w_T_root @ root_T_fl @ tf_matrix(jnp.array([0, -x[7], 0, 0, 0, 0]))
    w_T_flcon = w_T_fl @ fl_T_frcon
    return w_T_root, w_T_fr, w_T_frcon, w_T_fl, w_T_flcon


def kin_pj_right(x):
    w_T_root = tf_matrix_q(x[:7])
    w_T_fr = w_T_root @ root_T_fr @ tf_matrix(jnp.array([0, x[7], 0, 0, 0, 0]))
    w_T_frcon = w_T_fr @ fr_T_frcon
    return w_T_frcon


def kin_pj_left(x):
    w_T_root = tf_matrix_q(x[:7])
    w_T_fl = w_T_root @ root_T_fl @ tf_matrix(jnp.array([0, -x[7], 0, 0, 0, 0]))
    w_T_flcon = w_T_fl @ fl_T_frcon
    return w_T_flcon
