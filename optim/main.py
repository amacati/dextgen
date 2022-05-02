import re
import numpy as np
import time
import itertools

import nlopt
import jax.numpy as jnp
from jax import grad, jit, vmap, jacfwd
import jax.experimental.host_callback as hcb
from jax import random
from jax.config import config
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def myfunc(x, grad):
    if grad.size > 0:
        grad[:] = grad_fn(x, grad)
    return np.asarray(fn(x, grad)).item()  # JAX returns zero dim arrays instead of scalars


def fn(x, grad):
    return jnp.sqrt(x[1])


grad_fn = grad(fn)


def myconstraint(x, grad, a, b):
    if grad.size > 0:
        grad[:] = grad_c(x, None, a, b)
    return np.asarray(c(x, None, a, b)).item()


def c(x, _, a, b):
    return (a * x[0] + b)**3 - x[1]


grad_c = grad(c)


####################################################################################################
def objective(x, grad):
    if grad.size > 0:
        grad[:] = force_grad(x) + homogeneous_forces_grad(x)
    return np.asarray(force_reserve(x)).item() + np.asarray(homogeneous_force(x)).item()


@jit
def force_reserve(x):
    nc = len(x) // 6
    force_res = 0
    for i in range(nc):
        fc = x[i * 6 + 3:(i + 1) * 6]
        normal = -planes[contact_points[i]["side"]][0, 1]
        normf = jnp.linalg.norm(fc)
        fcnc = jnp.dot(fc, normal)
        fcnc /= (normf + 1e-9)  # If normf 0, sum is 0 anyways
        force_res -= normf * (alpha_max_t * fcnc - jnp.sqrt(1 - fcnc**2))
    return force_res


force_grad = jit(grad(force_reserve))


def force_constraints(result: np.ndarray, x: np.ndarray, grad: np.ndarray):
    if grad.size > 0:
        grad[:] = sum_of_forces_grad(x)
    result[:] = sum_of_forces(x)


@jit
def sum_of_forces(x):
    return jnp.sum(x.reshape(-1, 6)[:, 3:], axis=0)


sum_of_forces_grad = jit(jacfwd(sum_of_forces))


def moments_constraints(x: np.ndarray, grad: np.ndarray):
    if grad.size > 0:
        grad[:] = sum_of_moments_grad(x)
    return np.asarray(sum_of_moments(x)).item()


@jit
def sum_of_moments(x):
    x = x.reshape(-1, 6)
    return jnp.sum(jnp.cross(com - x[:, :3], x[:, 3:]))


sum_of_moments_grad = jit(grad(sum_of_moments))


def maximum_force_constraints(result: np.ndarray, x: np.ndarray, grad: np.ndarray):
    if grad.size > 0:
        grad[:] = force_norm_grad(x)
    result[:] = force_norm(x) - fmax


def minimum_force_constraints(result: np.ndarray, x: np.ndarray, grad: np.ndarray):
    if grad.size > 0:
        grad[:] = -force_norm_grad(x)
    result[:] = fmin - force_norm(x)


@jit
def force_norm(x):
    return jnp.linalg.norm(x.reshape(-1, 6)[:, 3:], axis=1)


force_norm_grad = jit(jacfwd(force_norm))


def homogeneous_forces_contraint(x: np.ndarray, grad: np.ndarray):
    if grad.size > 0:
        grad[:] = homogeneous_forces_grad(x)
    return np.asarray(homogeneous_force(x)).item()


@jit
def homogeneous_force(x):
    fnorm = force_norm(x)
    return jnp.sum((fnorm - jnp.mean(fnorm))**2)


homogeneous_forces_grad = jit(grad(homogeneous_force))


def generate_angle_constraint(i):
    _idx = i * 6

    @jit
    def force_angle_s(x):
        fc = x[_idx + 3:_idx + 6]
        normf = fc / jnp.linalg.norm(fc)
        return jnp.linalg.norm(jnp.cross(planes[contact_points[i]["side"]][0, 1], normf))

    force_angle_s_grad = jit(grad(force_angle_s))

    def angle_constraint(x, grad):
        angle = np.asarray(force_angle_s(x)).item()
        if grad.size > 0:
            if angle == 0:
                grad[:] = 0
            else:
                grad[:] = force_angle_s_grad(x)
        return angle - alpha_max_s

    return angle_constraint


def generate_plane_constraints(cp_idx, plane):
    assert plane.shape == (5, 2, 3)
    offsets = plane[:, 0]
    v_mat = np.zeros((5, 3))
    for i in range(5):
        normal = plane[i, 1]
        a_matrix = np.linalg.svd(np.cross(normal, np.identity(normal.shape[0]) * -1))[2]
        a_matrix[2, :] = normal / np.linalg.norm(normal)
        a_inv = np.linalg.inv(a_matrix.T)
        v_mat[i, :] = a_inv[2, :]

    @jit
    def plane_equality_constraint_jax(x):
        cp = x[cp_idx * 6:cp_idx * 6 + 3]
        return jnp.dot(v_mat[0, :], cp - offsets[0, :])

    plane_equality_constraint_grad = jit(grad(plane_equality_constraint_jax))

    def plane_equality_constraint(x, grad):
        if grad.size > 0:
            grad[:] = plane_equality_constraint_grad(x)
        return np.asarray(plane_equality_constraint_jax(x)).item()

    @jit
    def plane_inequality_constraints_jax(x):
        cp = x[cp_idx * 6:cp_idx * 6 + 3]
        return jnp.sum((cp - offsets[1:, ...]) * v_mat[1:, ...], axis=1)

    plane_inequality_constraints_grad = jit(jacfwd(plane_inequality_constraints_jax))

    def plane_inequality_constraints(result, x, grad):
        if grad.size > 0:
            grad[:] = plane_inequality_constraints_grad(x)
        result[:] = plane_inequality_constraints_jax(x)

    return plane_equality_constraint, plane_inequality_constraints


def distance_constraints(result: np.ndarray, x: np.ndarray, grad: np.ndarray):
    if grad.size > 0:
        grad[:] = distance_gradient(x)
    result[:] = distance(x)


@jit
def distance(x):
    pts = x.reshape(-1, 6)[:, :3]
    return min_dist - jnp.array(
        [jnp.linalg.norm(x[0] - x[1]) for x in itertools.combinations(pts, 2)])


distance_gradient = jit(jacfwd(distance))

if __name__ == "__main__":
    config.update("jax_enable_x64", True)
    config.update("jax_debug_nans", True)

    com = np.array([0, 0, 0])
    alpha_max = np.pi / 4
    alpha_max_s = np.sin(alpha_max)
    alpha_max_t = np.tan(alpha_max)
    fmax = 1
    fmin = 0.1 * fmax
    min_dist = 0.2

    cp0 = {"pos": np.array([1., 0.2, 0]), "f": np.array([-.75, 0, 0]), "side": 0}
    cp1 = {"pos": np.array([1., -0.2, 0]), "f": np.array([-.25, 0, 0]), "side": 0}
    cp2 = {"pos": np.array([-1., 0, 0]), "f": np.array([1., 0, 0]), "side": 1}
    contact_points = [cp0, cp1, cp2]
    ncp = len(contact_points)

    # Plane definition:
    # Surface plane origin, surface plane normal
    # Surface border0 origin, surface border0 normal
    # Surface border1 origin, surface border1 normal
    # Surface border2 origin, surface border2 normal
    # Surface border3 origin, surface border3 normal
    plane0 = np.array([[[1, 0, 0], [1, 0, 0]], [[1, 1, 0], [0, 1, 0]], [[1, -1, 0], [0, -1, 0]],
                       [[1, 0, 1], [0, 0, 1]], [[1, 0, -1], [0, 0, -1]]])

    plane1 = np.array([[[-1, 0, 0], [-1, 0, 0]], [[-1, 1, 0], [0, 1, 0]], [[-1, -1, 0], [0, -1, 0]],
                       [[-1, 0, 1], [0, 0, 1]], [[-1, 0, -1], [0, 0, -1]]])

    plane2 = np.array([[[0, 1, 0], [0, 1, 0]], [[1, 1, 0], [1, 0, 0]], [[-1, 1, 0], [-1, 0, 0]],
                       [[0, 1, 1], [0, 0, -1]], [[0, 1, -1], [0, 0, 1]]])

    plane3 = np.array([[[0, -1, 0], [0, -1, 0]], [[1, -1, 0], [1, 0, 0]], [[-1, -1, 0], [-1, 0, 0]],
                       [[0, -1, 1], [0, 0, -1]], [[0, -1, -1], [0, 0, 1]]])

    plane4 = np.array([[[0, 0, 1], [0, 0, 1]], [[1, 0, 1], [1, 0, 0]], [[-1, 0, 1], [-1, 0, 0]],
                       [[0, 1, 1], [0, 1, 0]], [[0, -1, 1], [0, -1, 0]]])

    plane5 = np.array([[[0, 0, -1], [0, 0, -1]], [[1, 0, -1], [1, 0, 0]], [[-1, 0, -1], [-1, 0, 0]],
                       [[0, 1, -1], [0, 1, 0]], [[0, -1, -1], [0, -1, 0]]])
    planes = [plane0, plane1, plane2, plane3, plane4, plane5]

    # Optimization
    xinit = np.concatenate([np.concatenate((cp["pos"], cp["f"])) for cp in contact_points])

    localopt = nlopt.opt(nlopt.LD_MMA, len(xinit))
    localopt.set_lower_bounds(-1)
    localopt.set_upper_bounds(1)
    localopt.set_xtol_rel(1e-3)

    opt = nlopt.opt(nlopt.AUGLAG, len(xinit))
    opt.set_xtol_rel(1e-3)
    opt.set_local_optimizer(localopt)
    opt.set_min_objective(objective)

    opt.add_inequality_constraint(generate_angle_constraint(0))
    opt.add_inequality_constraint(generate_angle_constraint(1))
    opt.add_inequality_constraint(generate_angle_constraint(2))

    for idx, cp in enumerate(contact_points):
        eq_constraint, ineq_constraints = generate_plane_constraints(idx, planes[cp["side"]])
        opt.add_equality_constraint(eq_constraint, 1e-6)
        opt.add_inequality_mconstraint(ineq_constraints, np.ones(4) * 1e-6)

    opt.add_equality_mconstraint(force_constraints, np.ones(3) * 1e-6)
    opt.add_equality_constraint(moments_constraints, 1e-6)
    opt.add_inequality_mconstraint(maximum_force_constraints, np.ones(ncp) * 1e-6)
    opt.add_inequality_mconstraint(minimum_force_constraints, np.ones(ncp) * 1e-6)
    opt.add_inequality_mconstraint(distance_constraints, np.ones(ncp * (ncp - 1) // 2) * 1e-6)
    #opt.add_equality_constraint(homogeneous_forces_contraint, 100)
    opt.set_lower_bounds(-3)
    opt.set_upper_bounds(3)

    tstart = time.perf_counter()
    xmin = opt.optimize(xinit)
    tend = time.perf_counter()
    print(f"Optimization took {tend-tstart:.2f}s")

    minf = opt.last_optimum_value()
    print("optimum at ", xmin)
    print("minimum value = ", minf)
    print("result code = ", opt.last_optimize_result())
    print(f"Sum of forces: {sum_of_forces(xmin)}")

    # Vizualization
    fig = plt.figure()
    fig.suptitle("Contact point optimization")
    ax = []
    ax.append(fig.add_subplot(121, projection="3d"))
    ax[0].set_title("Initial contact points")
    ax[0].set_box_aspect(aspect=(1, 1, 1))
    ax[0].set_xlim([-2, 2])
    ax[0].set_ylim([-2, 2])
    ax[0].set_zlim([-2, 2])
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_zlabel("z")

    ax.append(fig.add_subplot(122, projection="3d"))
    ax[1].set_title("Optimized contact points")
    ax[1].set_box_aspect(aspect=(1, 1, 1))
    ax[1].set_xlim([-2, 2])
    ax[1].set_ylim([-2, 2])
    ax[1].set_zlim([-2, 2])
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    ax[1].set_zlabel("z")

    for ax_id in range(2):
        r = [-1, 1]
        for s, e in itertools.combinations(np.array(list(itertools.product(r, r, r))), 2):
            if np.sum(np.abs(s - e)) == r[1] - r[0]:
                ax[ax_id].plot3D(*zip(s, e), color="k")
    for idx, contact_point in enumerate(contact_points):
        cp_pos = contact_point["pos"]
        force_pos = cp_pos + contact_point["f"]
        ax[0].scatter(cp_pos[0], cp_pos[1], cp_pos[2], color="r")
        ax[0].plot([cp_pos[0], force_pos[0]], [cp_pos[1], force_pos[1]], [cp_pos[2], force_pos[2]],
                   color="r")
        cp_pos = xmin[idx * 6:idx * 6 + 3]
        force_pos = cp_pos + xmin[idx * 6 + 3:(idx + 1) * 6]
        ax[1].scatter(*cp_pos, color="r")
        ax[1].plot([cp_pos[0], force_pos[0]], [cp_pos[1], force_pos[1]], [cp_pos[2], force_pos[2]],
                   color="r")

    plt.show()

    # opt = nlopt.opt(nlopt.LD_MMA, 2)
    # opt.set_lower_bounds([-float('inf'), 0])
    # opt.set_min_objective(myfunc)
    # opt.add_inequality_constraint(lambda x, grad: myconstraint(x, grad, 2, 0), 1e-8)
    # opt.add_inequality_constraint(lambda x, grad: myconstraint(x, grad, -1, 1), 1e-8)
    # opt.set_xtol_rel(1e-4)
    # x = np.array([1.234, 5.678])
    # x = opt.optimize(x)
    # minf = opt.last_optimum_value()
    # print("optimum at ", x[0], x[1])
    # print("minimum value = ", minf)
    # print("result code = ", opt.last_optimize_result())
