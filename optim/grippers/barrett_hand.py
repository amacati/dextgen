"""BarrettHand module."""
from __future__ import annotations
from typing import Dict, Callable

from optim.grippers.base_gripper import Gripper
from optim.grippers.kinematics.barrett_hand import BH_JOINT_LIMITS
from optim.utils.utils import import_guard

if import_guard():
    import numpy as np  # noqa: TC002, is guarded


class BarrettHand(Gripper):
    """Class to interface the BarrettHand kinematics with the optimization."""

    def __init__(self, info: Dict):
        """Initialize the gripper configuration.

        Args:
            info: Contact and gripper info dict.
        """
        super().__init__(info)

    @property
    def joint_limits(self) -> np.ndarray:
        """Joint limit property.

        Returns:
            The joint limits.
        """
        return BH_JOINT_LIMITS

    def create_kinematics(self, link: str, con_pt: Dict) -> Callable:
        """Define a function that calculates the link's contact point position.

        Args:
            link: The link name.

        Returns:
            A function that calculates the contact point position given the gripper configuration.
        """
        raise NotImplementedError

    def create_grasp_wrench(self, link: str) -> Callable:
        """Define the function to calculate the grasp wrench of the given link.

        Args:
            link: The link name.

        Returns:
            A function that calculates the grasp wrench.
        """
        raise NotImplementedError
