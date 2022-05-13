import numpy as np
import time
import itertools

import nlopt
from jax.config import config
import matplotlib.pyplot as plt

from constraints import generate_angle_constraint, generate_distance_constraints, force_constraints
from constraints import generate_maximum_force_constraints, generate_minimum_force_constraints
from constraints import generate_moments_constraints, generate_plane_constraints
from constraints import _sum_of_forces_jax
from objective import generate_objective
from geometry import generate_plane_normal, generate_plane_normals

if __name__ == "__main__":
    config.update("jax_enable_x64", True)
    config.update("jax_debug_nans", True)

    com = np.array([0, 0, 0])
    alpha_max = np.pi / 4
    max_angle_s = np.sin(alpha_max)
    max_angle_t = np.tan(alpha_max)
    max_angle_c = np.cos(alpha_max)
    fmax = 1
    fmin = 0.1 * fmax
    min_dist = 0.2

    cp0 = {"pos": np.array([1., 0.5, 0]), "f": np.array([-.75, -0.5, 0]), "side": 0}
    cp1 = {"pos": np.array([1., -0.5, 0]), "f": np.array([-.25, 0.5, 0]), "side": 0}
    cp2 = {"pos": np.array([-1., 0, 0.2]), "f": np.array([1., 0, 1]), "side": 1}
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
    normals = np.array([planes[cp["side"]][0, 1] for cp in contact_points])
    cp_normals = generate_plane_normals(normals)
    objective = generate_objective(cp_normals, max_angle_t)
    opt.set_min_objective(objective)

    for idx, cp in enumerate(contact_points):
        eq_constraint, ineq_constraints = generate_plane_constraints(idx, planes[cp["side"]])
        opt.add_equality_constraint(eq_constraint, 1e-6)
        opt.add_inequality_mconstraint(ineq_constraints, np.ones(4) * 1e-6)
        cp_normal = generate_plane_normal(planes[cp["side"]][0, 1])
        opt.add_inequality_constraint(generate_angle_constraint(idx, cp_normal, max_angle_c))

    opt.add_equality_mconstraint(force_constraints, np.ones(3) * 1e-6)
    opt.add_equality_mconstraint(generate_moments_constraints(com=com), np.ones(3) * 1e-6)
    opt.add_inequality_mconstraint(generate_minimum_force_constraints(fmin), np.ones(ncp) * 1e-6)
    opt.add_inequality_mconstraint(generate_maximum_force_constraints(fmax), np.ones(ncp) * 1e-6)
    opt.add_inequality_mconstraint(generate_distance_constraints(min_dist),
                                   np.ones(ncp * (ncp - 1) // 2) * 1e-6)
    # opt.add_equality_constraint(homogeneous_forces_contraint, 100)
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
    print(f"Sum of forces: {_sum_of_forces_jax(xmin)}")
    _x = xmin.reshape(-1, 6)
    sum_of_moments = np.sum(np.cross(com - _x[:, :3], _x[:, 3:]), axis=1)
    print(f"Sum of moments: {sum_of_moments}")

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
