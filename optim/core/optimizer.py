"""Optimizer module."""
import logging
from typing import Callable, Optional

from jax import jit
import jax.numpy as jnp
from jax.config import config
import numpy as np

from optim.core.interior_point import solve

logger = logging.getLogger(__name__)


class Optimizer:
    """Implementation of an interior-point optimizer in the style of NLOPT.

    Used to solve non-linear optimization problems with equality and inequality constraints.
    """

    def __init__(self, disable_nan_check: bool = False, use_single_precision: bool = False):
        """Initialize the optimizer with empty constraints and bounds.

        Args:
            disable_nan_check: Disable JAX nan check. Might improve computation speed.
            use_single_precision: Use jnp.float32 for calculations. Precision loss impacts
                optimization convergence.
        """
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

    def add_equality_constraint(self, f: Callable):
        """Add scalar equality constraint.

        Args:
            f: Equality function. Must be compatible with JAX autodiff.
        """
        self.equality_constraints.append(f)

    def add_equality_mconstraint(self, f: Callable):
        """Add vectorized equality constraints.

        Args:
            f: Equality functions. Must be compatible with JAX autodiff.
        """
        self.equality_mconstraints.append(f)

    def add_inequality_constraint(self, f: Callable):
        """Add scalar inequality constraint.

        Args:
            f: Inequality function. Inequality is expected in the form of c(x) >= 0. Must be
                compatible with JAX autodiff.
        """
        self.inequality_constraints.append(f)

    def add_inequality_mconstraint(self, f: Callable):
        """Add vectorized inequality constraints.

        Args:
            f: Inequality function. Inequalities are expected in the form of c(x) >= 0. Must be
                compatible with JAX autodiff.
        """
        self.inequality_mconstraints.append(f)

    def set_min_objective(self, f: Callable):
        """Set the optimization objective.

        Args:
            f: Objective function. Must be compatible with JAX autodiff.
        """
        self.objective = jit(f)

    def set_lower_bounds(self,
                         x: np.ndarray,
                         begin: Optional[int] = None,
                         end: Optional[int] = None):
        """Set lower bounds for the optimization variables.

        Args:
            x: The lwoer bounds.
            begin: Optional begin index of the variable bounds.
            end: Optional end index of the variable bounds.
        """
        self.xlow = np.array(x)
        if self._bounds_idx_low is not None and begin is not None and self._bounds_idx_low != begin:
            logger.warning("Redefinition of previously set lower bounds begin index.")
        if begin is not None:
            self._bounds_idx_low = begin
        if self._bounds_idx_high is not None and end is not None and self._bounds_idx_high != end:
            logger.warning("Redefinition of previously set lower bounds end index.")
        if end is not None:
            self._bounds_idx_high = end

    def set_upper_bounds(self,
                         x: np.ndarray,
                         begin: Optional[int] = None,
                         end: Optional[int] = None):
        """Set upper bounds for the optimization variables.

        Args:
            x: The upper bounds.
            begin: Optional begin index of the variable bounds.
            end: Optional end index of the variable bounds.
        """
        self.xup = np.array(x)
        if self._bounds_idx_low is not None and begin is not None and self._bounds_idx_low != begin:
            logger.warning("Redefinition of previously set upper bounds begin index.")
        if begin is not None:
            self._bounds_idx_low = begin
        if self._bounds_idx_high is not None and end is not None and self._bounds_idx_high != end:
            logger.warning("Redefinition of previously set upper bounds end index.")
        if end is not None:
            self._bounds_idx_high = end

    def set_maxeval(self, x: int):
        """Set the maximum number of allowed evaluations for the optimization.

        Args:
            x: Number of evaluations.
        """
        self.maxeval = x

    def optimize(self, xinit: np.ndarray, niter: Optional[int] = None) -> np.ndarray:
        """Optimize the objective with the registered constraints using an interior point method.

        Args:
            xinit: Optimization initialization.
            niter: Optional number of optimization iterations.

        Returns:
            The optimized variables.
        """
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
        """Reset the optimizer."""
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

    def _compile_equality_constraints(self) -> Callable:
        """Define and JIT the equality constraints.

        Note:
            JIT is invoked on first function evaluation, NOT on definition inside this function.

        Returns:
            The JITed equality constraints function.
        """
        eqs, meqs = self.equality_constraints.copy(), self.equality_mconstraints.copy()

        @jit
        def ce(x: jnp.ndarray) -> jnp.ndarray:
            """Equality constraints function.

            Args:
                x: Optimization variable input.

            Returns:
                The calculated equalities.
            """
            cnsts = jnp.array([f(x) for f in eqs])
            mcnsts = (f(x) for f in meqs)
            return jnp.concatenate((cnsts, *mcnsts))

        return ce

    def _compile_inequality_constraints(self, N: int) -> Callable:
        """Define and JIT the inequality constraints.

        Lower and upper bounds are included into the inequality constraints. If an index for the
        bounds is specified, only the indexed array is bounded.

        Note:
            JIT is invoked on first function evaluation, NOT on definition inside this function.

        Args:
            N: Dimension of x.

        Returns:
            The JITed inequality constraints function.
        """
        ineqs, mineqs = self.inequality_constraints.copy(), self.inequality_mconstraints.copy()
        bounds_idx = np.arange(self._bounds_idx_low or 0, self._bounds_idx_high or N)

        @jit
        def ci(x: jnp.ndarray) -> jnp.array:
            """Inequality constraints function.

            Args:
                x: Optimization variable input.

            Returns:
                The calculated inequalities.
            """
            cnsts = jnp.array([f(x) for f in ineqs])
            mcnsts = (f(x) for f in mineqs)
            cmin = x[bounds_idx] - self.xlow
            cmax = self.xup - x[bounds_idx]
            return jnp.concatenate((cnsts, *mcnsts, cmin, cmax))

        return ci
