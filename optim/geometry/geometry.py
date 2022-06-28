from typing import Dict

from optim.geometry.base import Geometry
from optim.geometry.cube import Cube
from optim.geometry.cylinder import Cylinder
from optim.geometry.sphere import Sphere


def get_object(info: Dict) -> Geometry:
    name = info["object_info"]["name"]
    if name == "cube":
        return Cube(info)
    elif name == "cylinder":
        return Cylinder(info)
    elif name == "sphere":
        return Sphere(info)
    raise RuntimeError(f"Unsupported object {name}")
