"""Interior point optimization module."""
import logging
import time
from typing import Callable, Tuple

from jax import jacfwd, jacrev, grad, jit
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

# jacfwd used forward differentiation mode, jacrev uses reverse-mode. Both are equal up to numerical
# precision, but jacfwd is faster for "tall" jacobians, jacrev is faster for "wide" jacobians. See
# https://github.com/google/jax/issues/47


def solve(f: Callable,
          ce: Callable,
          ci: Callable,
          x: np.ndarray,
          niter: int = 1000) -> Tuple[np.ndarray, int, int]:
    """Optimize a function with equality and inequality constraints with the interior point method.

    Callables have to be compatible with JAX autodiff.

    Returns:
        The optimized variables, the optimization status, and the number of required iterations.
    """
    logger.info("Interior point optimization routine called")
    nf, ni, ne = len(x), len(ci(x)), len(ce(x))
    nz = ni
    status = 0

    y = np.zeros(ne)
    z = np.zeros(nz)
    s = np.ones(ni)
    e = np.ones(ni)
    xall = np.concatenate((x, s, y, z))
    mu = 1.
    sigma = 0.2
    alpha_s_max = 1.

    @jit
    def lagrangian(x: np.ndarray, y: np.ndarray, z: np.ndarray, s: np.ndarray) -> float:
        """Calculate the lagrangian.

        Args:
            x: Optimization variables.
            y: Equality constraint lambdas.
            z: Inequality constraint lambdas.
            s: Inequality slack vector.

        Returns:
            The value of the lagrangian.
        """
        return f(x) - jnp.dot(y, ce(x)) - jnp.dot(z, (ci(x) - s))

    @jit
    def grad_lagrangian(xall: np.ndarray, mu: float) -> jnp.ndarray:
        """Calculate the gradient of the lagrangian.

        Args:
            xall: Complete variable vector with the lambdas of equality and inequality constraints.
            mu: Current slack of the inequality constraints.

        Returns:
            The gradient of the lagrangian.
        """
        x, s, y, z = xall[:nf], xall[nf:nf + ni], xall[nf + ni:nf + ne + ni], xall[nf + ne + ni:]
        Ae = jac_ce(x)
        Ai = jac_ci(x)
        df = grad_f(x)
        S = jnp.diag(s)
        return -jnp.concatenate((df - Ae.T @ y - Ai.T @ z, S @ z - mu * e, ce(x), ci(x) - s))

    hess_l = jit(jacfwd(jacrev(lagrangian, argnums=0), argnums=0))
    jac_ce = jit(jacfwd(ce))
    jac_ci = jit(jacfwd(ci))
    grad_f = jit(grad(f))

    N = nf + ni + ne + nz
    Mat = np.zeros((N, N))
    Mat[nf + ne + ni:, nf:nf + ni] = -np.eye(ni)
    vec = np.zeros(N)
    p = np.zeros(N)

    t0 = time.time()
    try:
        for i in range(niter):
            Ae = jac_ce(x)
            Ai = jac_ci(x)
            df = grad_f(x)
            Z = jnp.diag(z)
            S = jnp.diag(s)
            Hl = hess_l(x, y, z, s)
            if i == 0:
                logger.debug("JIT compile complete, starting optimization")

            Mat[:nf, :nf] = Hl
            Mat[:nf, nf + ni:nf + ne + ni] = -Ae.T
            Mat[:nf, nf + ne + ni:] = -Ai.T
            Mat[nf:nf + ni, nf:nf + ni] = Z
            Mat[nf:nf + ni, nf + ne + ni:] = S
            Mat[nf + ni:nf + ne + ni, :nf] = Ae
            Mat[nf + ne + ni:, :nf] = Ai

            vec[:nf] = -df + Ae.T @ y + Ai.T @ z
            vec[nf:nf + ni] = -S @ z + mu * e
            vec[nf + ni:nf + ne + ni] = -ce(x)
            vec[nf + ne + ni:] = -ci(x) + s

            sol = np.linalg.solve(Mat, vec)
            px = sol[:nf]
            ps = sol[nf:nf + ni]
            py = sol[nf + ni:nf + ne + ni]
            pz = sol[nf + ne + ni:]

            xall[:nf] = x
            xall[nf:nf + ni] = s
            xall[nf + ni:nf + ne + ni] = y
            xall[nf + ne + ni:] = z

            p[:nf] = px
            p[nf:nf + ni] = ps
            p[nf + ni:nf + ne + ni] = py
            p[nf + ne + ni:] = pz

            # Perform a linesearch on the norm of the gradient of the lagrangian to ensure a
            # suitable stepsize and a decrease of the objective function.
            xall[:], nfnew = linesearch(grad_lagrangian, xall, p, alpha_s_max, mu)

            x, s = xall[:nf], xall[nf:nf + ni]
            y, z = xall[nf + ni:nf + ne + ni], xall[nf + ne + ni:]

            if i % 100 == 0:
                logger.debug(f"Iteration {i}: dLnorm: {nfnew:.2e}")

            if nfnew < mu:
                mu = sigma * mu
            elif nfnew < 1e-7:
                logger.info(f"Optimization converged after {i+1} iterations")
                break
            if i == niter - 1:
                logger.warning(f"Optimization failed to converge after {i+1} iterations")
                status = -1
    except np.linalg.LinAlgError as e:
        logger.warning(f"Optimization error:\n{e}")
        status = -1
    logger.debug(f"Optimization took {time.time() - t0:.0f}s")
    return x, status, i


# We can't @jit linesearch because of the early stopping criteria
def linesearch(f: Callable, x0: np.ndarray, d: np.ndarray, alpha: float,
               mu: float) -> Tuple[np.ndarray, float]:
    """Linesearch to calculate a suitable step size.

    We can't JIT compile the linesearch because of the early stopping criteria (JAX functions can't
    contain control flows defined during runtime). Possibly circumvented by using Numba.

    Args:
        f: Target function for the linesearch.
        x0: Initial starting point variables.
        d: Search direction vector.
        alpha: Initial step size.
        mu: Allowed slack of inequality constraints.

    Returns:
        The updated variable and the new norm of the target function.
    """
    nf0 = jnp.linalg.norm(f(x0, mu))
    scale = 0.
    for _ in range(10):
        x1 = x0 + alpha / 2**scale * d
        nf = jnp.linalg.norm(f(x1, mu))
        if nf < nf0:
            return x1, nf
        scale += 1
    return x1, nf
