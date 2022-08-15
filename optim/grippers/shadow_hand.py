"""ShadowHand module."""
from typing import Dict, Callable

from optim.grippers.base_gripper import Gripper


class ShadowHand(Gripper):
    """Class to interface the ShadowHand kinematics with the optimization."""

    def __init__(self, info: Dict):
        """Initialize the gripper configuration.

        Args:
            info: Contact and gripper info dict.
        """
        super().__init__(info)

    @property
    def joint_limits(self):
        """Joint limit property.

        Returns:
            The joint limits.
        """
        raise NotImplementedError

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
