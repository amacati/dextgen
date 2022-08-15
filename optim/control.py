import logging
from typing import Dict

import numpy as np

from envs.rotations import embedding2mat, mat2quat, quat2mat
from optim.utils.utils import filter_info
from optim.core.optimizer import Optimizer
from optim.objective import create_cube_objective
from optim.geometry import get_geometry, Cube
from optim.grippers import get_gripper

logger = logging.getLogger(__name__)


class Controller:

    GRIPPER_EPS = 1e-2
    r_R_v = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])

    def __init__(self):
        self.opt = Optimizer()
        self.opt_pos = None
        self.opt_orient = None
        self.opt_grasp = None
        self._is_reached = False
        self._is_lowered = False
        self._is_grasped = False

    def reset(self):
        self._is_reached = False
        self._is_lowered = False
        self._is_grasped = False
        self.opt_pos = None
        self.opt_orient = None
        self.opt_grasp = None

    def __call__(self, state, goal):
        assert self.opt_grasp is not None
        robot_pos, w_R_r = state[:3], embedding2mat(state[3:9])
        gripper_state = state[9:11]
        geom_pos, w_R_g = state[14:17], embedding2mat(state[20:26])
        des_pos = geom_pos + self.opt_pos_rel
        w_R_rdes = w_R_g @ self.g_R_r
        reach_offset = np.array([0., 0, 0.055])
        if self._check_reached(robot_pos, des_pos + reach_offset, w_R_r, w_R_rdes):
            logger.debug("reaching is complete")
            self._is_reached = True
        if self._check_reached(robot_pos, des_pos, w_R_r, w_R_rdes):
            logger.debug("lowering is complete")
            self._is_lowered = True
        if not self._is_reached:
            logger.debug("reaching phase")
            pos_ctrl = self._compute_pos_ctrl(robot_pos, des_pos + reach_offset)
            gripper_ctrl = self.opt_grasp + self.GRIPPER_EPS
        elif not self._is_lowered:
            logger.debug("lowering phase")
            pos_ctrl = self._compute_pos_ctrl(robot_pos, des_pos)
            gripper_ctrl = self.opt_grasp + self.GRIPPER_EPS
        elif not self._is_grasped:
            logger.debug("grasping phase")
            pos_ctrl = self._compute_pos_ctrl(robot_pos, des_pos)
            gripper_ctrl = -np.ones(1)
            if np.mean(gripper_state) < 0.03:
                self._is_grasped = True
        else:
            logger.debug("goal reaching phase")
            pos_ctrl = self._compute_pos_ctrl(geom_pos, goal)
            gripper_ctrl = -np.ones(1)
        rot_ctrl = self._compute_rot_ctrl(w_R_g)
        return np.concatenate((pos_ctrl, rot_ctrl, gripper_ctrl))

    @staticmethod
    def _compute_pos_ctrl(pos, des_pos):
        dx = (des_pos - pos)
        dx = dx / (np.linalg.norm(dx) + 1e-2)
        return dx

    def _compute_rot_ctrl(self, w_R_g):
        w_R_r = w_R_g @ self.g_R_r
        w_R_v = w_R_r @ self.r_R_v
        rot_ctrl = w_R_v.flatten()
        return rot_ctrl

    def _check_reached(self, pos: np.ndarray, des_pos: np.ndarray, orient: np.ndarray,
                       des_orient: np.ndarray) -> bool:
        """Check if the desired pose has been reached given a fixed tolerance.

        Args:
            pos: Current position.
            des_pos: Desired position.
            orient: Current orientation as rotation matrix.
            des_orient: Desired orientation as rotation matrix.

        Returns:
            True if the pose is within tolerances, else False.
        """
        dx = pos - des_pos
        orient, des_orient = mat2quat(orient), mat2quat(des_orient)
        # Calculate the angle between the two orientations
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
        cube = get_geometry(info, gripper)
        self._check_geom(cube)
        cube.create_constraints(gripper, self.opt)
        gripper.create_constraints(self.opt)
        self.opt.set_min_objective(create_cube_objective(xinit, cube.com))
        xopt = self.opt.optimize(xinit, 10_000)
        self.opt_pos_rel = xopt[:3] - cube.pos
        w_R_r = quat2mat(xopt[3:7])
        w_R_g = cube.orient_mat
        self.g_R_r = w_R_g.T @ w_R_r
        # Convert from joint angle to interval of [-1, 1]
        self.opt_grasp = np.array([(xopt[7] + xopt[8]) / 0.05 * 2 - 1])
        if self.opt.status != 0:
            raise RuntimeError("Optimization failed to converge!")
        return xopt

    def _check_geom(self, cube: Cube):
        """Check if the object contacts are in a valid configuration for the optimization.

        Contacts have to lie on opposing sides of the cube since otherwise, the contact force
        constraints are violated.

        Args:
            cube: The cube object.

        Raises:
            RuntimeError: The contacts don't lie on opposing sides of the cube.
        """
        if abs(cube.contact_mapping[0] - cube.contact_mapping[1]) != 1:
            raise RuntimeError("Contacts don't lie on opposing sides of the cube")
