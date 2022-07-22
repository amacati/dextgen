import jax.numpy as jnp

from grippers.kinematics.parallel_jaw import kin_pj_right, kin_pj_left


def create_cube_objective(xinit, com):

    def objective(x):
        regularizer = jnp.sum((x - xinit)**2)
        diff_r, diff_l = kin_pj_right(x)[:3, 3] - com, kin_pj_left(x)[:3, 3] - com
        return regularizer * 1e-6 + jnp.sum(diff_r**2) + jnp.sum(diff_l**2)

    return objective
