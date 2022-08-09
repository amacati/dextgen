import logging

from jax import jit
import jax.numpy as jnp
from jax.config import config
import numpy as np

from optim.core.interior_point import solve

logger = logging.getLogger(__name__)


class Optimizer:
    # Implementation of an interior-point optimizer for non-linear optimization problems with
    # equality and inequality constraints

    def __init__(self, disable_nan_check=False, use_single_precision=False):
        self.objective = None
        self.equality_constraints = []
        self.equality_mconstraints = []
        self.inequality_constraints = []
        self.inequality_mconstraints = []
        self.status = 0
        self.niter = 0
        self.xlow = None
        self.xup = None
        self.maxeval = None
        self.last_optimum_value = None
        self._bounds_idx_low = None
        self._bounds_idx_high = None
        if not disable_nan_check:
            config.update("jax_debug_nans", True)
        if not use_single_precision:
            config.update("jax_enable_x64", True)

    def add_equality_constraint(self, f):
        self.equality_constraints.append(f)

    def add_equality_mconstraint(self, f):
        self.equality_mconstraints.append(f)

    def add_inequality_constraint(self, f):
        # Inequality constraints are expected in the form of c(x) >= 0
        self.inequality_constraints.append(f)

    def add_inequality_mconstraint(self, f):
        # Inequality constraints are expected in the form of c(x) >= 0
        self.inequality_mconstraints.append(f)

    def set_min_objective(self, f):
        self.objective = jit(f)

    def set_lower_bounds(self, x, begin=None, end=None):
        self.xlow = np.array(x)
        if self._bounds_idx_low is not None and begin is not None and self._bounds_idx_low != begin:
            logger.warning("Redefinition of previously set lower bounds begin index.")
        if begin is not None:
            self._bounds_idx_low = begin
        if self._bounds_idx_high is not None and end is not None and self._bounds_idx_high != end:
            logger.warning("Redefinition of previously set lower bounds end index.")
        if end is not None:
            self._bounds_idx_high = end

    def set_upper_bounds(self, x, begin=None, end=None):
        self.xup = np.array(x)
        if self._bounds_idx_low is not None and begin is not None and self._bounds_idx_low != begin:
            logger.warning("Redefinition of previously set upper bounds begin index.")
        if begin is not None:
            self._bounds_idx_low = begin
        if self._bounds_idx_high is not None and end is not None and self._bounds_idx_high != end:
            logger.warning("Redefinition of previously set upper bounds end index.")
        if end is not None:
            self._bounds_idx_high = end

    def set_maxeval(self, x):
        self.maxeval = x

    def optimize(self, xinit, niter=None):
        assert self.objective is not None
        assert self.status == 0
        assert self.equality_constraints or self.equality_mconstraints
        assert self.inequality_constraints or self.inequality_mconstraints
        assert self.xlow is not None and self.xup is not None
        niter = niter or self.maxeval or 10_000
        ce = self._compile_equality_constraints()
        ci = self._compile_inequality_constraints(len(xinit))
        xopt, status, niter = solve(self.objective, ce, ci, xinit, niter)
        self.status, self.niter = status, niter
        self.last_optimum_value = self.objective(xopt)
        return xopt

    def reset(self):
        self.objective = None
        self.equality_constraints = []
        self.equality_mconstraints = []
        self.inequality_constraints = []
        self.inequality_mconstraints = []
        self.status = 0
        self.xlow = None
        self.xup = None
        self.maxeval = None
        self.last_optimum_value = None
        self._bounds_idx_low = None
        self._bounds_idx_high = None

    def _compile_equality_constraints(self):
        eqs, meqs = self.equality_constraints.copy(), self.equality_mconstraints.copy()

        @jit
        def ce(x):
            cnsts = jnp.array([f(x) for f in eqs])
            mcnsts = (f(x) for f in meqs)
            return jnp.concatenate((cnsts, *mcnsts))

        return ce

    def _compile_inequality_constraints(self, N):
        ineqs, mineqs = self.inequality_constraints.copy(), self.inequality_mconstraints.copy()
        bounds_idx = np.arange(self._bounds_idx_low or 0, self._bounds_idx_high or N)

        @jit
        def ci(x):
            cnsts = jnp.array([f(x) for f in ineqs])
            mcnsts = (f(x) for f in mineqs)
            cmin = x[bounds_idx] - self.xlow
            cmax = self.xup - x[bounds_idx]
            return jnp.concatenate((cnsts, *mcnsts, cmin, cmax))

        return ci
