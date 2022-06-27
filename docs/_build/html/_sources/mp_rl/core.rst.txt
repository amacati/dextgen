.. _mp_rl.core:

mp_rl.core
==========
The ``mp_rl.core`` module contains the multiprocessing reinforcement learning algorithm implementation and the functionalities for auto-checkpointing and network loading.
Its main interface is the :ref:`mp_rl.core.ddpg` class which implements the training process and constructs the required :ref:`mp_rl.core.actor`, :ref:`mp_rl.core.critic`, :ref:`mp_rl.core.normalizer` and :ref:`mp_rl.core.replay_buffer`.
