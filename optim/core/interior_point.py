import logging
from jax import jacfwd, jacrev, grad, jit
import numpy as np
import jax.numpy as jnp
import time

logger = logging.getLogger(__name__)


def solve(f, ce, ci, x, niter=1000):
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
    def lagrangian(x, y, z, s):
        return f(x) - jnp.dot(y, ce(x)) - jnp.dot(z, (ci(x) - s))

    @jit
    def grad_lagrangian(xall, mu):
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

    logger.info("JIT derivatives created")

    N = nf + ni + ne + nz
    Mat = np.zeros((N, N))
    Mat[nf + ne + ni:, nf:nf + ni] = -np.eye(ni)
    vec = np.zeros(N)
    p = np.zeros(N)

    t0 = time.time()
    for i in range(niter):
        Ae = jac_ce(x)
        Ai = jac_ci(x)
        df = grad_f(x)
        Z = jnp.diag(z)
        S = jnp.diag(s)
        Hl = hess_l(x, y, z, s)
        if i == 0:
            logger.info("JIT compile complete, starting optimization")

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
        xall[:], nfnew = linesearch(grad_lagrangian, xall, p, alpha_s_max, mu)

        x, s, y, z = xall[:nf], xall[nf:nf + ni], xall[nf + ni:nf + ne + ni], xall[nf + ne + ni:]

        if nfnew < mu:
            mu = sigma * mu
        elif nfnew < 1e-12:
            logger.info(f"Optimization converged after {i+1} iterations")
            break
        if i == niter - 1:
            logger.warning(f"Optimization failed to converge after {i+1} iterations")
            status = -1
    logger.info(f"Optimization took {time.time() - t0:.0f}s")
    return x, status


# We can't @jit linesearch because of the early stopping criteria
def linesearch(f, x0, d, alpha, mu):
    nf0 = jnp.linalg.norm(f(x0, mu))
    scale = 0.
    for _ in range(10):
        x1 = x0 + alpha / 2**scale * d
        nf = jnp.linalg.norm(f(x1, mu))
        if nf < nf0:
            return x1, nf
        scale += 1
    return x1, nf
