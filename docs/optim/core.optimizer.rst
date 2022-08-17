.. _optim.core.optimizer:

core.optimizer
==============
The optimizer provides an object oriented interface to the :ref:`optim.core.interior_point` optimization routine. It fuses multiple inequality and equality constraints into single, vectorized constraint functions, adds variable bounds to the inequality constraints, and manages optimization settings. In order to provide a familiar interface, the optimizer mimics the API of `NLOPT <https://nlopt.readthedocs.io/en/latest/NLopt_Reference/>`_.

Since all constraints, objectives and lagrangians are written in Python, the speed of optimization can be extremely slow. Therefore, before running the optimization, we ``jit`` all functions and required derivatives. Initial compilation of the functions (and especially their derivatives) can take a significant amount of time, but makes the following optimization steps fast enough to enable reasonable convergence times.

.. note::
    Just in time compile times can reach up to minutes. It is currently not possible to save the compiled functions to a persistent function cache. Experimental support for TPUs is available in JAX, but has to be integrated into the optimizer first.

.. automodule:: optim.core.optimizer
    :members:
    :private-members:
