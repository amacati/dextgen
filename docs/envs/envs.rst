.. _envs.envs:

envs
====
The ``envs`` module contains the implementation of all learning environments. On module import, these environments are registered with OpenAI's ``gym`` module and become available via the ``gym.make(<env_name>)`` function. The environments are independent of the training implementation.

Module structure
~~~~~~~~~~~~~~~~
The ``assets`` folder contains the MJCF files that define the simulation environment in MuJoCo, as well as the required meshes for simulation. The assets are further split into the different gripper types.

The environments themselves are implemented as Python classes. Upon calling ``gym.make``, the environment class is initialized and acts as a wrapper around MuJoCo to control the environment logic. The experiments for different gripper types are further divided into separate submodules. Available gripper environment submodules are: :ref:`envs.parallel_jaw`, :ref:`envs.barrett_hand`, :ref:`envs.shadow_hand`. All environments use the pandas arm as a base for the gripper.

The :ref:`envs.utils` module contains various utility functions. Utilities related to rotation representation and conversion are located at :ref:`envs.rotations`. 

Credits
~~~~~~~
This module and its structure are based on the OpenAI robot environments available at the `Farama-Foundation repository <https://github.com/Farama-Foundation/Gym-Robotics>`_.
