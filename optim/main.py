import numpy as np

import nlopt
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.config import config


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

if __name__ == "__main__":
    config.update("jax_enable_x64", True)

    opt = nlopt.opt(nlopt.LD_MMA, 2)
    opt.set_lower_bounds([-float('inf'), 0])
    opt.set_min_objective(myfunc)
    opt.add_inequality_constraint(lambda x, grad: myconstraint(x, grad, 2, 0), 1e-8)
    opt.add_inequality_constraint(lambda x, grad: myconstraint(x, grad, -1, 1), 1e-8)
    opt.set_xtol_rel(1e-4)
    x = np.array([1.234, 5.678])
    x = opt.optimize(x)
    minf = opt.last_optimum_value()
    print("optimum at ", x[0], x[1])
    print("minimum value = ", minf)
    print("result code = ", opt.last_optimize_result())
