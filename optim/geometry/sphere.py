from typing import Dict

from optim.geometry.base_geometry import Geometry
from optim.grippers import Gripper


class Sphere(Geometry):

    def __init__(self, info: Dict):
        super().__init__(info)

    def create_surface_constraints(self, gripper: Gripper, opt):
        """Create constraints for contact points to remain on the geometry surface.

        Args:
            gripper: Optimization problem gripper.
            opt: Optimizer.
        """
        raise NotImplementedError
