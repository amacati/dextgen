.. Dext-Gen documentation master file.

Dext-Gen documentation
======================

Dext-Gen uses deep reinforcement learning with hindsight learning to achieve dexterous grasping in MuJoCo with different gripper types. It builds on OpenAI's gym.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started/setup
   getting_started/training
   getting_started/testing

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Environment Documentation

   envs/envs
   envs/parallel_jaw
   envs/barrett_hand
   envs/shadow_hand
   envs/seaclear
   envs/rotations
   envs/utils

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Learning Documentation

   mp_rl/core
   mp_rl/core.actor
   mp_rl/core.critic
   mp_rl/core.ddpg
   mp_rl/core.noise
   mp_rl/core.normalizer
   mp_rl/core.replay_buffer
   mp_rl/core.utils

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Optimization Documentation

   optim/optim
   optim/core
   optim/core.optimizer
   optim/core.interior_point
   optim/geometry
   optim/grippers
   optim/grippers.kinematics
   optim/objective
   optim/constraints
   optim/control
   optim/utils

.. toctree::
   :maxdepth: 1
   :caption: Notes

   notes/acknowledgements


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`