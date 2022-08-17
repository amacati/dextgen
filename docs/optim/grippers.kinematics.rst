.. _optim.grippers.kinematics:

Gripper kinematics
==================
Many operations in the optim module require the recalculation of the gripper kinematics with new configurations, and need to differentiate with respect to the configuration. The kinematic submodule replicates the kinematics from the used grippers with homogeneous transformation matrices and makes them differentiable with JAX.

.. note::
    Although all hands have been implemented, only the parallel jaw gripper has been tested for correctness.

.. warning::
    Gripper kinematics are hardcoded and do **not** adapt to changes in the robot configuration in the environment asset files!

kinematics.parallel_jaw
~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: optim.grippers.kinematics.parallel_jaw
    :members:

kinematics.barrett_hand
~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: optim.grippers.kinematics.barrett_hand
    :members:

kinematics.shadow_hand
~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: optim.grippers.kinematics.shadow_hand
    :members:

kinematics.tf
~~~~~~~~~~~~~
.. automodule:: optim.grippers.kinematics.tf
    :members:
