from typing import Callable
import nlopt
import numpy as np
import jax.numpy as jnp
from jax import grad, jacfwd
from jax.config import config
import jax.experimental.host_callback as hcb
from jax import make_jaxpr
from jax import lax

import random
import matplotlib.pyplot as plt
import time
from numba import njit


def normalize(x):
    return x / np.linalg.norm(x)


def generate_objective_fn(normals) -> Callable:

    def objective(x: np.ndarray, grad: np.ndarray) -> float:
        if grad.size > 0:
            grad[:] = _jax_d_objective(x, normals)
        c = np.asarray(_jax_objective(x, normals)).item()  # JAX returns scalars as 0D arrays
        assert not np.isnan(c)
        return c

    return objective


def _jax_objective(x: jnp.ndarray, normals: jnp.ndarray) -> jnp.ndarray:
    # cost = 0
    # x = x.reshape(-1, 3)  # Turn 1D array of parameters into 2D array of Nx3 points
    # for i in range(len(x)):
    #     for j in range(len(x)):
    #         if i == j:
    #             continue
    #         d = (x[j] - x[i])
    #         norm_d = jnp.linalg.norm(d)
    #         d /= norm_d
    #         dxn = jnp.cross(d, normals[i])
    #         cost += 0.5 * jnp.linalg.norm(dxn) ** 2
    p1 = x[:3]
    p2 = x[3:6]
    p3 = x[6:9]
    connect12 = jnp.cross(n1, (p2 - p1) / jnp.linalg.norm(p2 - p1))
    connect21 = jnp.cross(n2, (p2 - p1) / jnp.linalg.norm(p2 - p1))
    connect13 = jnp.cross(n1, (p3 - p1) / jnp.linalg.norm(p3 - p1))
    connect31 = jnp.cross(n3, (p1 - p3) / jnp.linalg.norm(p1 - p3))
    connect23 = jnp.cross(n2, (p3 - p2) / jnp.linalg.norm(p3 - p2))
    connect32 = jnp.cross(n3, (p2 - p3) / jnp.linalg.norm(p2 - p3))

    return jnp.linalg.norm(connect12)**2 + jnp.linalg.norm(connect21)**2 + jnp.linalg.norm(
        connect13)**2 + jnp.linalg.norm(connect31)**2 + jnp.linalg.norm(
            connect23)**2 + jnp.linalg.norm(connect32)**2


_jax_d_objective = grad(_jax_objective)


def generate_plane_constraints(normals, ds):

    def c(result: np.ndarray, x: np.ndarray, grad: np.ndarray):
        result[:] = _jax_plane_constraints(x, normals, ds)
        if grad.size > 0:
            grad[:] = _jax_jac_plane_constraints(x, normals, ds)
        assert not any(np.isnan(result))

    return c


def _jax_plane_constraints(x, normals, ds):
    return (jnp.sum(x.reshape(-1, 3) * normals, axis=1) + ds)**2


_jax_jac_plane_constraints = jacfwd(_jax_plane_constraints)


def generate_distance_constraints(npoints, dst):
    i_idx = np.array([i for i in range(npoints) for _ in range(i + 1, npoints)])
    j_idx = np.array([j for i in range(npoints) for j in range(i + 1, npoints)])

    def c(result: np.ndarray, x: np.ndarray, grad: np.ndarray):
        result[:] = _jax_distance_constraints(x, i_idx, j_idx, dst)
        if grad.size > 0:
            grad[:] = _jax_jac_distance_constraints(x, i_idx, j_idx, dst)
        assert not any(np.isnan(result))

    return c


def _jax_distance_constraints(x, i_idx, j_idx, dst):
    x = jnp.reshape(x, (-1, 3))
    dx = x[i_idx, :] - x[j_idx, :]
    return -jnp.linalg.norm(dx, axis=1) + dst


_jax_jac_distance_constraints = jacfwd(_jax_distance_constraints)


def generate_boundary_constraints(Linv, l_coefs, l_centers):

    def c(result: np.ndarray, x: np.ndarray, grad: np.ndarray):
        result[:] = _jax_boundary_constraints(x, Linv, l_coefs, l_centers)
        if grad.size > 0:
            grad[:] = _jax_jac_boundary_constraints(x, Linv, l_coefs, l_centers)
        assert not any(np.isnan(result))

    return c


def _jax_boundary_constraints(x, Linv, l_coefs, l_centers):
    x = jnp.reshape(x, (-1, 3))
    #hcb.id_print([1]*10)
    #hcb.id_print(x)
    x = x - l_centers
    coefs = jnp.einsum("ijk,ik->ij", Linv, x)
    #hcb.id_print(coefs)
    #hcb.id_print(l_centers)
    return jnp.reshape(abs(coefs) - l_coefs, (-1,))


_jax_jac_boundary_constraints = jacfwd(_jax_boundary_constraints)

if __name__ == "__main__":
    config.update("jax_enable_x64", True)
    config.update("jax_debug_nans", True)

    com = np.array([0., 0., 0.])
    n1 = normalize(np.array([0., -1, 0.]))
    n2 = normalize(np.array([0., 1., 0.]))
    n3 = normalize(np.array([0., 1., 0.]))

    d1 = np.array([0., -1, 0.])
    d2 = np.array([0., 1., 0.])
    d3 = np.array([0., 1., 0.])

    ly = np.array([[1., 0.], [0., 0.], [0., 1.]])
    L = np.array((ly, ly, ly))
    Linv = np.array([np.linalg.pinv(l) for l in L])
    l_coefs = np.array([[1., 0.], [1., 0.], [1., 0.]])
    l_centers = np.vstack((d1, d2, d3))

    p1 = np.array([1.5, -1.1, 0.])
    p2 = np.array([1.4, 1.2, 0.])
    p3 = np.array([1.6, 1.3, 0.])

    x = np.vstack((p1, p2, p3))
    normals = np.vstack((n1, n2, n3))
    _ds = np.vstack((d1, d2, d3))
    ds = -np.sum(normals * _ds, axis=1)

    fig, ax = plt.subplots(1, 2)
    ax[0].set_xlim([-2, 2])
    ax[0].set_ylim([-2, 2])
    ax[0].set_aspect("equal")
    ax[1].set_xlim([-2, 2])
    ax[1].set_ylim([-2, 2])
    ax[1].set_aspect("equal")

    ax[0].plot([-1, 1], [-1, -1], color="k")
    ax[0].plot([-1, 1], [1, 1], color="k")
    ax[0].plot([-1, -1], [-1, 1], color="k")
    ax[0].plot([1, 1], [-1, 1], color="k")
    ax[0].scatter(x[:, 0], x[:, 1], color="r")
    for i in range(len(x)):
        ax[0].plot([x[i, 0], x[i, 0] + .3 * normals[i, 0]], [x[i, 1], x[i, 1] + .3 * normals[i, 1]],
                   color="c")

    f = generate_objective_fn(normals)
    opt = nlopt.opt(nlopt.LD_AUGLAG_EQ, len(x.flatten()))

    opt.set_upper_bounds(3)
    opt.set_lower_bounds(-3)
    opt.set_min_objective(f)
    opt.set_xtol_rel(1e-4)

    opt.add_inequality_mconstraint(generate_distance_constraints(len(x), 1e-2),
                                   np.ones(int(len(x) * (len(x) - 1) / 2)) * 1e-2)
    opt.add_equality_mconstraint(generate_plane_constraints(normals, ds), np.ones(3) * 1e-6)
    opt.add_inequality_mconstraint(generate_boundary_constraints(Linv, l_coefs, l_centers),
                                   np.ones(3 * 2) * 1e-6)

    print(f"Initial value: {f(x.flatten(), np.empty(0)):.3e}")
    xsol = opt.optimize(x.flatten())
    print(f"Minimal value: {opt.last_optimum_value():.3e}")
    xsol = xsol.reshape((-1, 3))
    print(xsol)
    ax[1].plot([-1, 1], [-1, -1], color="k")
    ax[1].plot([-1, 1], [1, 1], color="k")
    ax[1].plot([-1, -1], [-1, 1], color="k")
    ax[1].plot([1, 1], [-1, 1], color="k")
    ax[1].scatter(xsol[:, 0], xsol[:, 1], color="r")
    for i in range(len(x)):
        ax[1].plot([xsol[i, 0], xsol[i, 0] + .3 * normals[i, 0]],
                   [xsol[i, 1], xsol[i, 1] + .3 * normals[i, 1]],
                   color="c")
    plt.show()
