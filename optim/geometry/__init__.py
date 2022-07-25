from typing import Dict

from optim.geometry.base_geometry import Geometry
from optim.geometry.cube import Cube
from optim.geometry.cylinder import Cylinder
from optim.geometry.sphere import Sphere
from optim.grippers.base_gripper import Gripper


def get_geometry(info: Dict, gripper: Gripper = None) -> Geometry:
    name = info["object_info"]["name"]
    if name == "cube":
        assert gripper is not None
        return Cube(info, gripper)
    elif name == "cylinder":
        return Cylinder(info)
    elif name == "sphere":
        return Sphere(info)
    raise RuntimeError(f"Unsupported geometry {name}")
