"""Controller module."""
import logging
from typing import Dict, Tuple

import numpy as np

from envs.rotations import embedding2mat, mat2quat, quat2mat
from optim.utils.utils import filter_info
from optim.core.optimizer import Optimizer
from optim.objective import create_cube_objective
from optim.geometry import get_geometry, Cube
from optim.grippers import get_gripper

logger = logging.getLogger(__name__)


class Controller:
    """Controller class for controlling the agent with the optimized grasp configuration."""

    GRIPPER_EPS = 1e-3
    r_T_v = np.array([[0., 1., 0., 0.], [0., 0., 1., 0.], [1., 0., 0., 0.], [0., 0., 0., 1.]])

    def __init__(self):
        """Initialize the controller and its optimizer."""
        self.opt = Optimizer()
        self.opt_pos = None
        self.opt_orient = None
        self.opt_grasp = None
        self._is_reached = False
        self._is_lowered = False
        self._is_grasped = False
        self.geom = None

    def reset(self):
        """Reset the controller."""
        self._is_reached = False
        self._is_lowered = False
        self._is_grasped = False
        self.opt_pos = None
        self.opt_orient = None
        self.opt_grasp = None
        self.opt.reset()

    def __call__(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Compute the control input for the environment given the current state and goal.

        Args:
            state: Current (unnormalized) environment state.
            goal: Current (unnormalized) environment goal.

        Returns:
            The control vector.

        Raises:
            AssertionError: No optimal grasp is registered.
        """
        assert self.opt_grasp is not None, "No optimal grasp registered"
        gripper_state = state[9:11]
        w_T_r = np.eye(4)  # Robot transformation
        w_T_r[:3, :3] = embedding2mat(state[3:9])
        w_T_r[:3, 3] = state[:3]
        w_T_g = np.eye(4)  # Geometry transformation
        w_T_g[:3, :3] = embedding2mat(state[20:26])
        w_T_g[:3, 3] = state[14:17]
        w_T_rdes = w_T_g @ self.g_T_r
        reach_offset = np.zeros((4, 4))
        reach_offset[:3, 3] = [0., 0, 0.055]
        if self._check_reached(w_T_r, w_T_rdes + reach_offset):
            logger.debug("reaching is complete")
            self._is_reached = True
        if self._check_reached(w_T_r, w_T_rdes):
            logger.debug("lowering is complete")
            self._is_lowered = True
        if not self._is_reached:
            logger.debug("reaching phase")
            pos_ctrl, rot_ctrl = self._compute_ctrl(w_T_r, w_T_rdes + reach_offset)
            gripper_ctrl = self.opt_grasp + self.GRIPPER_EPS
        elif not self._is_lowered:
            logger.debug("lowering phase")
            pos_ctrl, rot_ctrl = self._compute_ctrl(w_T_r, w_T_rdes)
            gripper_ctrl = self.opt_grasp + self.GRIPPER_EPS
        elif not self._is_grasped:
            logger.debug("grasping phase")
            pos_ctrl, rot_ctrl = self._compute_ctrl(w_T_r, w_T_rdes)
            gripper_ctrl = -np.ones(1)
            if np.mean(gripper_state) < 0.03:
                self._is_grasped = True
        else:
            logger.debug("goal reaching phase")
            w_T_goal = w_T_rdes.copy()
            w_T_goal[:3, 3] = goal
            pos_ctrl, rot_ctrl = self._compute_ctrl(w_T_g, w_T_goal)
            gripper_ctrl = -np.ones(1)
        return np.concatenate((pos_ctrl, rot_ctrl, gripper_ctrl))

    def _compute_ctrl(self, w_T_r: np.ndarray, w_T_d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the pose control input given the current pose and a target pose.

        Args:
            w_T_r: Current pose as homogeneous transformation matrix.
            w_T_d: Desired pose as homogeneous transformation matrix.

        Returns:
            The position control and the orientation control as flattened rotation matrix.
        """
        dx = (w_T_d[:3, 3] - w_T_r[:3, 3])
        dx = dx / (np.linalg.norm(dx) + 1e-2)
        dr = (w_T_d @ self.r_T_v)[:3, :3]
        return dx, dr.flatten()

    def _check_reached(self, w_T_r: np.ndarray, w_T_d: np.ndarray) -> bool:
        """Check if the current pose is within target tolerances of the desired pose.

        Args:
            w_T_r: Current pose as homogeneous transformation matrix.
            w_T_d: Desired pose as homogeneous transformation matrix.

        Returns:
            True if the pose is within tolerances, else False.
        """
        dx = w_T_r[:3, 3] - w_T_d[:3, 3]
        orient, des_orient = mat2quat(w_T_r[:3, :3]), mat2quat(w_T_d[:3, :3])
        dq = 2 * np.arccos(np.clip(np.abs(np.sum(orient * des_orient, axis=-1)), -1, 1))
        pos_reached, orient_reached = np.linalg.norm(dx) < 5e-3, dq < 1e-2
        return pos_reached and orient_reached

    def optimize_grasp(self, info: Dict) -> np.ndarray:
        """Optimize the grasp configuration and set it as the current target for the controller.

        Args:
            info: Contact information dictionary.

        Returns:
            The optimized configuration.

        Raises:
            RuntimeError: The optimization has failed to converge.
        """
        xinit, info = filter_info(info)
        self.opt.reset()
        gripper = get_gripper(info)
        self.geom = get_geometry(info, gripper)
        self._check_geom(self.geom)
        self.geom.create_constraints(gripper, self.opt)
        gripper.create_constraints(self.opt)
        self.opt.set_min_objective(create_cube_objective(xinit, self.geom.com))
        xopt = self.opt.optimize(xinit, 10_000)
        w_T_r = np.zeros((4, 4))
        w_T_r[:3, :3] = quat2mat(xopt[3:7])
        w_T_r[:4, 3] = np.concatenate((xopt[:3], np.array([1])))
        w_T_g = np.zeros((4, 4))
        w_T_g[:3, :3] = self.geom.orient_mat
        w_T_g[:4, 3] = np.concatenate((self.geom.pos, np.array([1])))
        self.g_T_r = np.linalg.inv(w_T_g) @ w_T_r
        self.opt_grasp = np.array([(xopt[7] + xopt[8]) / 0.05 * 2 - 1])
        if self.opt.status != 0:
            raise RuntimeError("Optimization failed to converge!")
        return xopt

    def set_geom(self, info: Dict):
        """Set the current controller target geometry.

        Args:
            info: Contact information dictionary.
        """
        gripper = get_gripper(info)
        self.geom = get_geometry(info, gripper)

    def set_xopt(self, xopt: np.ndarray):
        """Set the current solution of the controller.

        Warning:
            The controller target geometry has to be set first with `set_geom`!

        Args:
            xopt: Optimal gripper configuration.
        """
        assert self.geom is not None, "Geometry has to be set to set xopt"
        w_T_r = np.zeros((4, 4))
        w_T_r[:3, :3] = quat2mat(xopt[3:7])
        w_T_r[:4, 3] = np.concatenate((xopt[:3], np.array([1])))
        w_T_g = np.zeros((4, 4))
        w_T_g[:3, :3] = self.geom.orient_mat
        w_T_g[:4, 3] = np.concatenate((self.geom.pos, np.array([1])))
        self.g_T_r = np.linalg.inv(w_T_g) @ w_T_r
        self.opt_grasp = np.array([(xopt[7] + xopt[8]) / 0.05 * 2 - 1])

    def _check_geom(self, geom: Cube):
        """Check if the object contacts are in a valid configuration for the optimization.

        Contacts have to lie on opposing sides of the cube since otherwise, the contact force
        constraints are violated.

        Args:
            geom: The cube object.

        Raises:
            RuntimeError: The contacts don't lie on opposing sides of the cube.
        """
        if abs(geom.contact_mapping[0] - geom.contact_mapping[1]) != 1:
            raise RuntimeError("Contacts don't lie on opposing sides of the cube")
