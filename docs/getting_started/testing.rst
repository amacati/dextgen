.. _testing:

Testing
=======

Testing trained agents
~~~~~~~~~~~~~~~~~~~~~~
After successful training, the agents can be tested and visually inspected by executing 

.. code-block:: bash

   $ python test.py --env <env_name>

, where <env_name> is replaced with the environment name. The saved agent is loaded from the latest save in the environment save directory under ``saves/``. For example, when loading the parallel jaw experiment with a cube target object, the agent save located under ``saves/FlatPJCube-v0/`` is used. You can disable the rendering of environments with the ``--render n`` option, and control the amount of tests with the ``--ntests`` argument.

.. note::
    Without a trained agent, you cannot run the tests.

.. note::
    The agent is expected to be consistent with the current configuration in ``config/experiment_config.yaml``!

Testing the packages
~~~~~~~~~~~~~~~~~~~~
In order to test the functionality of environments and their compatibility with the training process, you can run the test suite by executing

.. code-block:: bash

   $ pytest tests

from the root directory. This should provide you with a summary of successful and failed tests. Note that environments have to be included in the ``envs.available_envs`` list of available environments from the ``envs.__init__`` file to be included in the test suite.
