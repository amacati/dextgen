.. _envs.shadow_hand:

envs.shadow_hand
================
This module contains all environments that are implemented with the ShadowHand. The available environments are:

``FlatSHCube-v0``, ``FlatSHCylinder-v0``, ``FlatSHSphere-v0``, ``FlatSHMesh-v0``, ``FlatSHAll-v0``, ``UnevenSHCube-v0``, ``UnevenSHMesh-v0``, ``ObstacleSHCube-v0``, ``FlatSHOrient-v0``.

Flat environments place the object on a flat surface. Uneven environments feature a work surface with uneven ground so that objects assume more diverse initial poses. Orientation environments feature full object pose goals. Obstacle environments include an additional forbidden region that the agent is forbidden to cross. 