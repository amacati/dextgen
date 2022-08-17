.. _optim.constraints:

Constraint functions
====================
Constraints should be registered with the object oriented interfaces provided for the gripper and the target object. The ``constraints`` submodule contains the function factories that parameterize and return the underlying constraint functions. All constraints have to be JAX differentiable. All inequality constraints are defined as

.. math::
    c(x) \geq 0.

It is worth emphasizing that the functions, with the exception of the quaternion constraint, **return functions**, and **not** the constraint values themselves. To calculate the constraints, the newly created functions have to be called with the robot configuration. All constraints take exactly one argument, the robot configuration, as their input.

.. note::
    Some factories return single constraints, vectorized constraints, or a mix of equality and inequality constraints. Refer to the individual function docs for specific information about the returned functions.

optim.constraints
~~~~~~~~~~~~~~~~~
.. automodule:: optim.constraints
    :members: