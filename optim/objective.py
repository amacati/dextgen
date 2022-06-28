import numpy as np
import jax.numpy as jnp
from jax import jit, grad
from constraints import _homogeneous_force_jax, _homogeneous_force_jax_grad


def create_objective(cp_normals, max_angle_t):

    def objective(x, grad):
        if grad.size > 0:
            grad[:] = force_grad(x) + _homogeneous_force_jax_grad(x)
        return np.asarray(force_reserve(x)).item() + np.asarray(_homogeneous_force_jax(x)).item()

    @jit
    def force_reserve(x):
        nc = len(x) // 6
        force_res = 0
        normals = cp_normals(x)
        for i in range(nc):
            fc = x[i * 6 + 3:(i + 1) * 6]
            normal = -normals[i, :]  # Negative for correct alignment with forces
            normf = jnp.linalg.norm(fc)
            fcnc = jnp.dot(fc, normal)
            fcnc /= (normf + 1e-9)  # If normf 0, sum is 0 anyways
            force_res -= normf * (max_angle_t * fcnc - jnp.sqrt(1 - fcnc**2))
        return force_res

    force_grad = jit(grad(force_reserve))

    return objective
