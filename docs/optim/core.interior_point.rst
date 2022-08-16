.. _optim.core.interior_point:

core.interior_point
===================
In order to solve the grasp optimization problem, this module uses a standard interior point method. It accepts an objective function, one vectorized equality constraint containing all equality constraints, one vectorized inequality constraint containing all inequality constraints, and solves the problem using the lagrangian of the optimization. After calculating the best descend direction with the associated hessian, a linesearch ensures a recrease of the objective function.

.. automodule:: optim.core.interior_point
    :members:
