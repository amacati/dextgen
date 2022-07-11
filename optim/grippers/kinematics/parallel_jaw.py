import numpy as np
from jax import jit
import jax.numpy as jnp
from optim.grippers.kinematics.tf import tf_matrix, tf_matrix_q

PJ_JOINT_LIMITS = {"lower": np.array([0]), "upper": np.array([0.05])}
JOINT_RANGE = 0.05


def pj_kinematics(link):
    return _kinematics_left if link == "robot0:l_gripper_finger_link" else _kinematics_right


@jit
def pj_full_kinematics(x):
    w_T_root = tf_matrix_q(x[:7])
    root_T_right = tf_matrix(np.array([0., 0.0159, 0.1, 0., np.pi / 2, 0]))
    root_T_left = tf_matrix(np.array([0., -0.0159, 0.1, 0., np.pi / 2, 0]))
    w_T_fr = w_T_root @ root_T_right @ tf_matrix(jnp.array([0, x[7], 0, 0, 0, 0]))
    w_T_fl = w_T_root @ root_T_left @ tf_matrix(jnp.array([0, -x[7], 0, 0, 0, 0]))
    return w_T_root, w_T_fr, w_T_fl


@jit
def _kinematics_right(x):
    w_T_root = tf_matrix_q(x[:7])
    root_T_right = tf_matrix(np.array([0., 0.0159, 0.1, 0., np.pi / 2, 0]))
    return w_T_root @ root_T_right @ tf_matrix(jnp.array([0, 0, x[7], 0, 0, 0]))


@jit
def _kinematics_left(x):
    w_T_root = tf_matrix_q(x[:7])
    root_T_left = tf_matrix(np.array([0., -0.0159, 0.1, 0., np.pi / 2, 0]))
    return w_T_root @ root_T_left @ tf_matrix(jnp.array([0, 0, -x[7], 0, 0, 0]))
