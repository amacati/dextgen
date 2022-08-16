.. _optim:

optim
=====
The ``optim`` module contains the grasp optimization process for optimizing the contact points of grasps obtained by running a trained reinforcement learning agent. The optimizer and the optimization routine are contained in :ref:`optim.core`. The :ref:`optim.grippers` submodule provides an interface to the used grippers. Likewise, :ref:`optim.geometry` is an interface to the target objects used in the environments.

.. warning::
    The ``optim`` package is still experimental. While the interfaces are generally designed to be easily extendable to other grippers and objects, the current version **only** supports the parallel jaw gripper with the cube target object.

How does it work?
~~~~~~~~~~~~~~~~~
The optim test loads a trained agent and simulates a grasping episode until it detects a valid grasp for the optimization. The observed robot configuration gets saved, the environment is reset to its starting point and the optimizer tries to optimize the robot configuration with respect to the defined grasp quality metric under the given grasp constraints. After finding a solution to the problem, a controller is invoked that deterministically tries to reach the optimized grasp pose and lifts the target object into its goal.

Getting started 
~~~~~~~~~~~~~~~
In order to test the optimization scheme, you can run the test script from the root directory with

.. code-block:: bash

   $ python optim/test.py --env FlatPJCube-v0

Details on how the agents is loaded are included in the :ref:`interface section <optim_interface>`. You can disable the rendering of environments with the ``--render n`` option, and control the amount of tests with the ``--ntests`` argument.

.. note::
    Without a trained agent, you cannot run the optimization tests.

.. _optim_interface:
Interfacing with the learned agents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The saved agent is loaded from the latest save in the environment save directory under ``saves/``. For the parallel jaw experiment with a cube target object, this is the agent save located under ``saves/FlatPJCube-v0/``. Since the optimization requires a good initial configuration guess, a successful grasp run is required before the configuration can be optimized. Agents therefore need to have learned a highly successful grasp strategy.

.. note::
    The agent is expected to be consistent with the current configuration in ``config/experiment_config.yaml``!

Both the robot and target object information is translated into the optimization from the contact information dictionary provided by the environment. This means that the optimization is only indirectly coupled with the simulation description. 

.. warning:: 
    In particular, the gripper kinematics are replicated according to the current robot description and do **not** change with the simulation description in the MuJoCo files!

Module design
~~~~~~~~~~~~~
The :ref:`optim.core` module provides an optimizer object largely consistent with the API of the `NLOPT package <https://nlopt.readthedocs.io/en/latest/NLopt_Reference/>`_, albeit with reduced functionality. This enables an optimization routine based on second order interior point methods instead of the gradient based methods of the NLOPT package. Its implementation in Python facilitates debugging of convergence failures and constraints.

All functions in the ``optim`` package that are part of the constraints or objective function are differentiable through `JAX <https://jax.readthedocs.io/en/latest/notebooks/quickstart.html>`_. JAX is an automatic differentiation library that clones numpy's functionality, but allows for automatic differentiation under a few conditions. Simultaneously, it allows for a just in time compilation of JAX-compatible functions to significantly improve computation speed.