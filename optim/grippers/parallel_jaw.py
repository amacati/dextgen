"""ParallelJaw gripper module."""
from __future__ import annotations
from typing import Dict, Callable

import numpy as np
import jax.numpy as jnp

from optim.grippers.base_gripper import Gripper
from optim.grippers.kinematics.parallel_jaw import PJ_JOINT_LIMITS, kin_pj_right, kin_pj_left
from optim.utils.utils import import_guard

if import_guard():
    from optim.core.optimizer import Optimizer  # noqa: TC001, is guarded


class ParallelJaw(Gripper):
    """Class to interface the ParallelJaw gripper kinematics with the optimization."""

    KP = 30_000
    LINKS = ("robot0:r_gripper_finger_link", "robot0:l_gripper_finger_link")

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
        return PJ_JOINT_LIMITS

    def create_kinematics(self, link: str, _: Dict) -> Callable:
        """Define a function that calculates the finger tip position.

        Args:
            link: The link name.

        Returns:
            A function that calculates the finger tip position given the gripper configuration.
        """
        assert link[7] in ["r", "l"]
        kin_finger = kin_pj_right if link[7] == "r" else kin_pj_left

        def kinematics(x: jnp.ndarray) -> jnp.ndarray:
            """Calculate the finger position of a gripper finger tip.

            Args:
                x: The gripper configuration.

            Returns:
                The finger tip position.
            """
            return kin_finger(x)[:3, 3]

        return kinematics

    def create_grasp_wrench(self, link: str) -> Callable:
        """Define the function to calculate the grasp wrench of the given link.

        Args:
            link: The link name.

        Warning:
            Currently unused, has to be adopted to new optimization variable definition. If used,
            the optimization variable additionally has to contain the next joint positions to
            calculate the grasp wrench with the P-controller of MuJoCo.

        Returns:
            A function that calculates the grasp wrench.
        """
        assert link in self.LINKS
        kin_finger = kin_pj_right if link[0] == "r" else kin_pj_left
        link_axis = 1 if link == self.LINKS[0] else -1  # left gripper moves along -y axis
        xidx = -1 if link == self.LINKS[0] else -2

        def grasp_wrench(x: jnp.ndarray) -> jnp.ndarray:
            """Calculate the grasp wrench of a finger.

            Args:
                x: The gripper configuration.

            Returns:
                The gripper wrench.
            """
            frame = kin_finger(x)
            f = frame[:3, 1] * link_axis * self.KP * (-x[xidx])
            return jnp.array([*f, 0, 0, 0])  # Wrench with zero torque elements

        return grasp_wrench

    def create_gripper_constraints(self, opt: Optimizer):
        """Create and add a constraint to couple the gripper fingers.

        Args:
            opt: Optimizer.
        """

        def pj_finger_constraint(x: jnp.ndarray) -> float:
            """Calculate the squared difference of the finger joint positions.

            Args:
                x: The gripper configuration.

            Returns:
                The squared difference of joint positions.
            """
            return (x[-1] - x[-2])**2

        opt.add_equality_constraint(pj_finger_constraint)
