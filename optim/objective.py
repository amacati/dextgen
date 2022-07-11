import numpy as np
import jax.numpy as jnp
from jax import jacfwd, jacrev, jit, grad
# from constraints import _homogeneous_force_jax, _homogeneous_force_jax_grad

global optim_it
optim_it = 1


def create_objective(cp_normals, grasp_forces, max_angle_t):

    def objective(x, grad):
        global optim_it
        if not optim_it % 100:
            print(optim_it)
        optim_it += 1
        if grad.size > 0:
            grad[:] = force_grad(x)
        f_reserve = np.asarray(force_reserve(x)).item()
        return f_reserve

    @jit
    def force_reserve(x):
        normals = cp_normals(x)
        f = grasp_forces(x)[:, :3]
        f_norm = jnp.linalg.norm(f, axis=1)
        # Normals point outwards, forces inwards. To align axis, inverse normals
        f_cos = jnp.sum(f * -normals, axis=1)  # Vectorized scalar product
        fn_cos = f_cos / (f_norm + 1e-9)  # Addition for numerical stability. If normf 0, _fcnc is 0
        return jnp.sum(f_norm * (max_angle_t * fn_cos - jnp.sqrt(1 - fn_cos**2)))

    force_grad = jit(grad(force_reserve))

    return objective


def create_objective_2(gripper, plane1, plane2):
    kinematics = gripper.create_kinematics(gripper.con_links[0], None)
    kinematics2 = gripper.create_kinematics(gripper.con_links[1], None)
    offsets1 = plane1[:, 0]
    v_mat1 = plane1[:, 1]
    offsets2 = plane2[:, 0]
    v_mat2 = plane2[:, 1]

    def objective(x):
        assert x.shape == (11,)
        f = jnp.sum(x[:9]**2)
        g1 = jnp.dot(kinematics(x) - offsets1[0, :], v_mat1[0, :])
        g2 = jnp.dot(kinematics2(x) - offsets2[0, :], v_mat2[0, :])
        return jnp.array(f + x[10] * g1 + x[11] * g2)

    hessian = jacfwd(jacrev(objective))

    return objective, hessian


def create_objective_3(gripper, plane1):
    kinematics = gripper.create_kinematics(gripper.con_links[0], None)
    offsets1 = plane1[:, 0]
    v_mat1 = plane1[:, 1]

    def lagrangian(x):
        assert x.shape == (10,)
        f = jnp.sum(x[:9]**2)
        g1 = jnp.dot(kinematics(x) - offsets1[0, :], v_mat1[0, :])
        return jnp.array(f + x[9] * g1)

    _grad = grad(lagrangian)

    hessian = jacfwd(grad(lagrangian))

    return lagrangian, _grad, hessian
