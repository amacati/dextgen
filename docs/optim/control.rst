.. _optim.control:

Controller
==========
All ``optim`` components are integrated into the ``Controller`` class, which takes a contact information dictionary, creates the gripper and geometry, optimizes the gripper configuration, and executes a scripted policy. This policy controls the gripper into the desired grasp configuration, grasps, and lifts the object into the target.

.. note::
    The scripted policy is a prototype and has not been optimized. Significant improvements in success rates are likely achievable by tuning the process.

optim.control
~~~~~~~~~~~~~
.. automodule:: optim.control
    :members:
    :special-members:
    :private-members:
