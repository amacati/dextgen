.. _training:

Training
========
After installing the package and all its dependencies, you can run the simulations and train agents to solve the different grasping scenarios.
You can expect training speeds to vary from a few hours to days depending on your hyperparameter configuration, hardware setup and task difficulty.

Configuring experiments
~~~~~~~~~~~~~~~~~~~~~~~
All experiments can be configured in the ``config/experiment_config.yaml`` file. The top ``Default`` key contains default settings that are applied to all experiments.
To add changes for a single environment, you can add the environment's name as a key to the config file. Settings under the explicit environment name overwrite the default values.
For some environments such as the ShadowHand simulations, additional parameters can be passed to the environment on initialization. These parameters should be included in the
individual experiment configuration under the ``kwargs`` key.

.. note::
    For reproducible experiments, do **not** set the ``seed`` argument to 0, as this seed will be ignored.

Running an experiment
~~~~~~~~~~~~~~~~~~~~~
All experiments are started by launching ``main.py`` with ``mpirun`` and the environment's name as argument. As an example, to launch the ShadowHand mesh grasp experiment,
the following command has to be executed.

.. code-block:: bash

   $ mpirun -n 16 python main.py --env FlatSHMesh-v0


You can control the number of MPI nodes with the ``-n`` argument. The environment name is passed with the ``--env`` agument. A list of the available environments can be found in the ``__init__.py`` file of the ``envs`` module.
Be sure to configure your experiment before launching the training nodes.

If you do not have access to 16 physical cores but only 16 hardware threads, you can launch MPI with the hardware thread option enabled.

.. code-block:: bash

   $ mpirun -n 16 --use-hwthread-cpus python main.py --env FlatSHMesh-v0


.. note::
    Changing the number of MPI nodes will result in different computation times and training performance. We recommend the use of 16 nodes in all experiments.

.. warning::
    Do **not** use python to launch the training. Doing so will not launch the required nodes and result in vastly different training results and computation times.


Please make sure the appropriate conda environment has been activated and all necessary environment variables are set (see :ref:`setup`).


Results
~~~~~~~
During training the results are saved to the saves folder in the root directory. The saves are further categorized into folders by their environment name.
The top level of an environment's save folder contains the latest network and normalizer checkpoints, as well as the current training progress plot and the current training statistics as a JSON file for convenient access during training.
The ``stats.json`` file also contains the configuration parameters of the current experiment. If several environments are trained concurrently, saves from different experiments overwrite each other. For this reason, each experiment also saves checkpoints
of the training progress to a folder under ``backup`` named after the date and time at the experiment start. These backup saves only contain the training statistics, not the full networks. The final networks and normalizers are saved to the backup folder once training has finished.

Pre-training
~~~~~~~~~~~~
Instead of training from scratch, networks can be initialized by previously trained agents. To enable network loading, use the ``load_pretrained`` option in the experiment configuration.
Pretrained networks are loaded from the ``saves/pretrain`` folder. If a folder exists that matches the name of the environment directly, networks are loaded from this directory. Otherwise loading falls to the networks saved under ``saves/pretrain/<gripper_name>``. Available gripper names are
``ParallelJaw``, ``BarrettHand`` and ``ShadowHand``.

In order to successfully load a pretrained agent, the pretrain save directory must contain four files, namely the actor network ``actor.pt``, the critic network ``critic.pt``,
the state normalizer ``state_norm.pkl`` and the goal normalizer ``goal_norm.pkl``.

Notes on distributed training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We use MPI to average the network gradients across the nodes. In general, ``torch.distributed`` and in particular ``torch.nn.parallel.DistributedDataParallel`` was developed to take care of this task.
However, we saw significant speed differences between torch's Gloo backend and manually managing the gradients with MPI. In theory, torch can be built with MPI as backend, however we opted for keeping the installation as
easy as possible. Furthermore, since the employed neural nets are small in total size, there is no speed gain in moving the network calculations to a GPU. If deeper nets are desired, it is recommended to revisit the
decision to train on CPUs only and move to GPUs. If possible, this should go hand in hand with a switch to torch's ``DDP`` with ``NCCL`` backend.