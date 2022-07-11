from typing import Dict

from optim.geometry import Geometry
from optim.grippers import Gripper


class Cylinder(Geometry):

    def __init__(self, info: Dict):
        super().__init__(info)
        self.contact_mapping = self._contact_mapping()

    def _contact_mapping(self):
        raise NotImplementedError

    def create_normals(self):
        return NotImplementedError

    def create_surface_constraints(self, gripper: Gripper, opt):
        for idx, con_pt in enumerate(self.con_pts):
            if self.contact_mapping[idx] == 0:
                # Create upper disk constraint
                ...
            elif self.contact_mapping[idx] == 1:
                # Create lower disk constraint
                ...
            else:
                # Create lateral surface constraint
                ...
