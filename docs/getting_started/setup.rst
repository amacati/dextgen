.. _setup:

Setup
=====
The following section describes how to install the necessary packages for running the experiments. We assume a functional installation of
conda or miniconda running on Linux with at least 16 hardware threads.

Mujoco
~~~~~~
All environments are simulated in Mujoco. We therefore require a functional installation of Mujoco 2.1. Mujoco has recently been made open source and can be installed from the official `github repository <https://github.com/deepmind/mujoco/releases>`_.

Python environment
~~~~~~~~~~~~~~~~~~
All required packages are included in the packages ``environment.yaml`` file. It is recommended to create a new conda environment from this file in order to install the dependencies. The training process requires an installation of MPI. 

.. note::
    Although conda includes the required OpenMPI binaries, you might run into issues if you have a prior installation of MPI on your system.

.. warning::
    The ``Dockerfile`` included in the repository is used for CI purposes only. Training via Docker has not been tested and is not recommended.

After installing the environment, make sure to activate it and run 

.. code-block:: bash

   $ pip install .

from the repositories root folder to install the dext-gen packages.

Common issues
~~~~~~~~~~~~~
If Mujoco is missing required environment variables, make sure to include the environment variable exports into your ``.bashrc`` file by replacing ``<user>`` with your user name.

.. code-block:: bash

   $ echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/<user>/.mujoco/mujoco210/bin' >> ~/.bashrc
   $ echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia' >> ~/.bashrc

If the environment install fails, check that you have installed ``patchelf`` and ``libosmesa``. You can install the libraries with

.. code-block:: bash

   $ sudo apt install patchelf
   $ sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
