.. _optim.grippers:

Grippers
========
The grippers module is a collection of the grippers used in the experiments of the main package. Each gripper is derived from the :class:`optim.grippers.base_gripper.Gripper` base class. Grippers extract the gripper configuration from the simulation and have to generate functions that calculate the grasp forces and contact point positions given a gripper configuration.

.. note::
    At the moment, only the ParallelJaw gripper is implemented.

grippers.base_gripper
~~~~~~~~~~~~~~~~~~~~~
.. automodule:: optim.grippers.base_gripper
    :members:
    :private-members:

grippers.parallel_jaw
~~~~~~~~~~~~~~~~~~~~~
.. automodule:: optim.grippers.parallel_jaw
    :members:
    :private-members:

grippers.barrett_hand
~~~~~~~~~~~~~~~~~~~~~
.. automodule:: optim.grippers.barrett_hand
    :members:
    :private-members:

grippers.shadow_hand
~~~~~~~~~~~~~~~~~~~~
.. automodule:: optim.grippers.shadow_hand
    :members:
    :private-members:
