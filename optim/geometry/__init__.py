"""The Geometry module provides an interface for the target objects in the grasp optimization."""
from __future__ import annotations
from typing import Dict

from optim.geometry.cube import Cube
from optim.utils.utils import import_guard

if import_guard():
    from optim.grippers.base_gripper import Gripper  # noqa: TC001, is guarded
    from optim.geometry.base_geometry import Geometry  # noqa: TC001, is guarded


def get_geometry(info: Dict, gripper: Gripper = None) -> Geometry:
    """Get the correct geometry class from the contact information.

    Args:
        info: Contact info directory.
        gripper: Corresponding gripper.

    Returns:
        The correct geometry object.

    Raises:
        RuntimeError: The contact info contains an unknown geometry type.
    """
    name = info["object_info"]["name"]
    if name == "cube":
        assert gripper is not None
        return Cube(info, gripper)
    raise RuntimeError(f"Unsupported geometry {name}")
