.. _optim.geometry:

Grasp target objects
====================
Target objects from the environments are part of the grasp constraints. The ``geometry`` module defines :class:`optim.geometry.base_geometry.Geometry` as the common base class for all object interfaces. Its child classes read the information from the environment contact dictionary, initialize the object pose and enable the addition of object specific constraints to the optimization.

.. note::
    At the moment, only the Cube geometry is implemented.

geometry.base_geometry
~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: optim.geometry.base_geometry
    :members:
    :private-members:

geometry.cube
~~~~~~~~~~~~~
.. automodule:: optim.geometry.cube
    :members:
    :private-members:
