"""Gripper base module."""
from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
from typing import Dict, Callable

import numpy as np

from envs.rotations import mat2quat

from optim.constraints import quaternion_cnst
from optim.utils.utils import import_guard

if import_guard():
    from optim.core.optimizer import Optimizer  # noqa: TC001, is guarded


class Gripper(ABC):
    """Base class for grippers that interface the gripper kinematics with the optimization."""

    def __init__(self, info: Dict):
        """Initialize the gripper configuration.

        Args:
            info: Contact and gripper info dict.
        """
        self.pos = np.array(info["gripper_info"]["pos"])
        self.orient_q = mat2quat(np.array(info["gripper_info"]["orient"]))
        self.grip_state = np.array(info["gripper_info"]["state"])
        self.state = np.concatenate((self.pos, self.orient_q, self.grip_state))
        self.con_links = np.array([i["geom1"] for i in info["contact_info"]])

    @abstractmethod
    def create_kinematics(self, con_pt: Dict) -> Callable:
        """Define a function that calculates the link's contact point position.

        Args:
            link: The link name.

        Returns:
            A function that calculates the contact point position given the gripper configuration.
        """

    @abstractmethod
    def create_gripper_constraints(self, opt: Optimizer) -> Callable:
        """Create and add gripper specific constraints to the optimizer.

        Args:
            opt: Optimizer.
        """

    @abstractproperty
    def joint_limits(self) -> np.ndarray:
        """Joint limit property.

        Returns:
            The joint limits.
        """

    def create_constraints(self, opt: Optimizer):
        """Create the gripper configuration constraints.

        Args:
            opt: Optimizer.
        """
        # Unit quaternion constraint for orientation
        opt.add_equality_constraint(quaternion_cnst)
        self.create_gripper_constraints(opt)
        low, high = self.joint_limits["lower"], self.joint_limits["upper"]
        opt.set_lower_bounds(low, begin=7)
        opt.set_upper_bounds(high, begin=7)
