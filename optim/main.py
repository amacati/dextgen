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
    nc = len(x) // 4
    force_res = 0
    for i in range(nc):
        fc = x[i * 4 + 2:i * 4 + 4]
        normal = sides[contact_points[i]["side"]]["normal"]
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
    return jnp.sum(x.reshape(-1, 4)[:, 2:], axis=0)


sum_of_forces_grad = jit(jacfwd(sum_of_forces))


def moments_constraints(x: np.ndarray, grad: np.ndarray):
    if grad.size > 0:
        grad[:] = sum_of_moments_grad(x)
    return np.asarray(sum_of_moments(x)).item()


@jit
def sum_of_moments(x):
    x = x.reshape(-1, 4)
    return jnp.sum(jnp.cross(com - x[:, :2], x[:, 2:]))


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
    return jnp.linalg.norm(x.reshape(-1, 4)[:, 2:], axis=1)


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
    _idx = i * 4

    @jit
    def force_angle_s(x):
        fc = x[_idx + 2:_idx + 4]
        normf = fc / jnp.linalg.norm(fc)
        crossp = jnp.cross(sides[contact_points[i]["side"]]["normal"], normf)
        return crossp

    force_angle_s_grad = jit(grad(force_angle_s))

    def angle_constraint(x, grad):
        angle = np.asarray(force_angle_s(x)).item()
        if grad.size > 0:
            grad[:] = force_angle_s_grad(x)
        return angle - alpha_max_s

    return angle_constraint


def plane_constraints(result: np.ndarray, x: np.ndarray, grad: np.ndarray):
    if grad.size > 0:
        grad[:] = plane_equality_grad(x)
    result[:] = plane_equality(x)


@jit
def plane_equality(x):  # TODO: Calculate from sides and normals
    return jnp.array([x[1] + 1, x[5] - 1, x[9] - 1])


plane_equality_grad = jit(jacfwd(plane_equality))


def distance_constraints(result: np.ndarray, x: np.ndarray, grad: np.ndarray):
    if grad.size > 0:
        grad[:] = distance_gradient(x)
    result[:] = distance(x)


@jit
def distance(x):
    pts = x.reshape(-1, 4)[:, :2]
    return min_dist - jnp.array(
        [jnp.linalg.norm(x[0] - x[1]) for x in itertools.combinations(pts, 2)])


distance_gradient = jit(jacfwd(distance))

if __name__ == "__main__":
    config.update("jax_enable_x64", True)
    config.update("jax_debug_nans", True)

    com = np.array([0, 0])
    alpha_max = np.pi / 4
    alpha_max_s = np.sin(alpha_max)
    alpha_max_t = np.tan(alpha_max)
    fmax = 1
    fmin = 0.1 * fmax
    min_dist = 0.2

    cp0 = {"pos": np.array([0, -1]), "f": np.array([0, 1]), "side": 0}
    cp1 = {"pos": np.array([0.0, 1]), "f": np.array([0, -0.75]), "side": 2}
    cp2 = {"pos": np.array([-0.0, 1]), "f": np.array([0, -0.25]), "side": 2}
    contact_points = [cp0, cp1, cp2]
    ncp = len(contact_points)

    s0 = {
        "center": np.array([0, -1]),
        "normal": np.array([0, 1]),
        "basis": np.array([1, 0]),
        "len": 2
    }
    s1 = {
        "center": np.array([1, 0]),
        "normal": np.array([-1, 0]),
        "basis": np.array([0, 1]),
        "len": 2
    }
    s2 = {
        "center": np.array([0, 1]),
        "normal": np.array([0, -1]),
        "basis": np.array([1, 0]),
        "len": 2
    }
    s3 = {
        "center": np.array([-1, 0]),
        "normal": np.array([1, 0]),
        "basis": np.array([0, 1]),
        "len": 2
    }
    sides = [s0, s1, s2, s3]

    # Optimization
    xinit = np.concatenate([np.concatenate((cp["pos"], cp["f"])) for cp in contact_points])

    pts = xinit.reshape(-1, 4)[:, :2]
    pts2 = pts.reshape(pts.shape[0], 1, pts.shape[1])

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
    opt.add_equality_mconstraint(force_constraints, np.ones(2) * 1e-6)
    opt.add_equality_constraint(moments_constraints, 1e-6)
    opt.add_equality_mconstraint(plane_constraints, np.ones(3) * 1e-6)
    opt.add_inequality_mconstraint(maximum_force_constraints, np.ones(3) * 1e-6)
    opt.add_inequality_mconstraint(minimum_force_constraints, np.ones(3) * 1e-6)
    #opt.add_inequality_mconstraint(distance_constraints, np.ones(ncp*(ncp-1)//2)*1e-6)
    #opt.add_equality_constraint(homogeneous_forces_contraint, 100)
    opt.set_lower_bounds(-1)
    opt.set_upper_bounds(1)

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
    fig, ax = plt.subplots(1, 2)
    ax[0].set_xlim([-2, 2])
    ax[0].set_ylim([-2, 2])
    ax[0].set_aspect("equal")
    ax[1].set_xlim([-2, 2])
    ax[1].set_ylim([-2, 2])
    ax[1].set_aspect("equal")
    for side in sides:
        xstart = side["center"] - 0.5 * side["len"] * side["basis"]
        xend = side["center"] + 0.5 * side["len"] * side["basis"]
        ax[0].plot([xstart[0], xend[0]], [xstart[1], xend[1]], color="k")
        ax[1].plot([xstart[0], xend[0]], [xstart[1], xend[1]], color="k")
    for idx, contact_point in enumerate(contact_points):
        cp_pos = contact_point["pos"]
        force_pos = cp_pos + contact_point["f"]
        ax[0].scatter(cp_pos[0], cp_pos[1], color="r")
        ax[0].plot([cp_pos[0], force_pos[0]], [cp_pos[1], force_pos[1]], color="r")
        cp_pos = xmin[idx * 4:idx * 4 + 2]
        force_pos = cp_pos + xmin[idx * 4 + 2:idx * 4 + 4]
        ax[1].scatter(cp_pos[0], cp_pos[1], color="r")
        ax[1].plot([cp_pos[0], force_pos[0]], [cp_pos[1], force_pos[1]], color="r")

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
