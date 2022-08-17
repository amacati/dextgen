.. _envs.seaclear:

envs.seaclear
==============
The SeaClear environment simulates the SeaClear underwater robot with its scooping gripper. In addition to the standard position goal, a forbidden region is included that results in a reward of -2 whenever the target object is closer than 0.15m, and leads to a failed episode if the target object has entered the zone at any point. The SeaClear environment name is ``SeaClear-v0``.

.. automodule:: envs.seaclear
