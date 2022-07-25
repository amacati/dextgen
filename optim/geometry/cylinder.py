"""Cylinder geometry module for grasp optimization."""
from typing import Dict
from optim.core.optimizer import Optimizer

from optim.geometry import Geometry
from optim.grippers import Gripper


class Cylinder(Geometry):
    """Class to automate constraint generation for grasp optimization on cylinder objects."""

    def __init__(self, info: Dict):
        """Initialize a cylinder geometry.

        Args:
            info: Contact info directory.
        """
        super().__init__(info)
        self.contact_mapping = self._contact_mapping()

    def _contact_mapping(self):
        raise NotImplementedError

    def create_surface_constraints(self, gripper: Gripper, opt: Optimizer):
        """Create constraints for contact points to remain on the geometry surface.

        Args:
            gripper: Optimization problem gripper.
            opt: Optimizer.
        """
        raise NotImplementedError
