from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

from optim.visualization import visualize_contacts, visualize_geometry, visualize_gripper

from envs.rotations import embedding2quat, quat2mat
from optim.utils import filter_info
from optim.core.optimizer import Optimizer
from optim.objective import create_cube_objective
from optim.geometry import get_geometry
from optim.grippers import get_gripper


class Controller:

    GRIPPER_EPS = 1e-2

    def __init__(self):
        self.opt = Optimizer()
        self.opt_pos = None
        self.opt_orient = None
        self.opt_grasp = None
        self._is_reached = False
        self._is_grasped = False

    def reset(self):
        self._is_reached = False
        self._is_grasped = False
        self.opt_pos = None
        self.opt_orient = None
        self.opt_grasp = None

    def __call__(self, state, goal):
        assert self.opt_grasp is not None
        robot_pos, robot_orientq = state[:3], embedding2quat(state[3:9])
        dx = np.linalg.norm(robot_pos - self.opt_pos)
        dq = np.linalg.norm(self.opt_orient - robot_orientq)
        if dx < 0.01 and dq < 0.1:
            self._is_reached = True
        if not self._is_reached:
            pos_ctrl = self._compute_pos_ctrl(robot_pos, self.opt_pos)
            gripper_ctrl = self.opt_grasp + self.GRIPPER_EPS
        elif not self._is_grasped:
            pos_ctrl = self._compute_pos_ctrl(robot_pos, self.opt_pos)
            gripper_ctrl = np.zeros(1)
            self._is_grasped = True
        else:
            pos_ctrl = self._compute_pos_ctrl(robot_pos, goal)
            gripper_ctrl = np.zeros(1)
        return np.concatenate((pos_ctrl, quat2mat(self.opt_orient).flatten(), gripper_ctrl))

    @staticmethod
    def _compute_pos_ctrl(pos, goal):
        return goal - pos

    def optimize_grasp(self, info: Dict):
        xinit, info = filter_info(info)
        self.opt.reset()
        geom = get_geometry(info)
        gripper = get_gripper(info)
        geom.create_constraints(gripper, self.opt)
        gripper.create_constraints(self.opt)
        fig = visualize_geometry(geom)
        fig = visualize_gripper(gripper, fig)
        fig = visualize_contacts(geom.con_pts, fig)
        plt.show()
        self.opt.set_min_objective(create_cube_objective(xinit, geom.com))
        xopt = self.opt.optimize(xinit, 1000)
        self.opt_pos = xopt[:3]
        self.opt_orient = xopt[3:7]
        # Convert from joint angle to interval of [-1, 1]
        self.opt_grasp = np.array([xopt[7] / 0.05 * 2 - 1])
        if self.opt.status != 0:
            raise RuntimeError("Optimization failed to converge!")
