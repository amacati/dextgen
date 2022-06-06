.. RLGrasps documentation master file.

RLGrasps documentation
====================================

RLGrasps is a project to learn dexterous grasping in MuJoCo with different gripper types. It builds on OpenAI's gym.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started/setup
   getting_started/training

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Environment Documentation

   envs/envs
   envs/barrett
   envs/parallel_jaw
   envs/shadow_hand
   envs/seaclear
   envs/rotations
   envs/utils

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Learning Documentation

   mp_rl/mp_rl
   mp_rl/core.actor
   mp_rl/core.critic
   mp_rl/core.ddpg
   mp_rl/core.noise
   mp_rl/core.normalizer
   mp_rl/core.replay_buffer
   mp_rl/core.utils

.. toctree::
   :maxdepth: 1
   :caption: Notes

   notes/acknowledgements


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`