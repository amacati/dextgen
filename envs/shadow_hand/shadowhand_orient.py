"""ShadowHandOrientation class file."""
from typing import Optional

from gym import utils
import numpy as np

import envs
from envs.shadow_hand.shadowhand_base import ShadowHandBase


class ShadowHandOrientation(ShadowHandBase, utils.EzPickle):
    """Environment for pick and place with the ShadowHand and free hand orientation."""

    def __init__(self,
                 reward_type: str = "sparse",
                 n_eigengrasps: Optional[int] = None,
                 p_grasp_start: float = 0.):
        """Initialize the Mujoco sim.

        Params:
            reward_type: Choice of reward formular.
            n_eigengrasps: Number of eigengrasp vectors the agent gets as action input.
            p_grasp_start: Fraction of episode starts with pregrasped objects.
        """
        n_actions = 7 + (n_eigengrasps or 20)
        super().__init__(n_actions=n_actions,
                         reward_type=reward_type,
                         p_grasp_start=p_grasp_start,
                         n_eigengrasps=n_eigengrasps)
        utils.EzPickle.__init__(self,
                                reward_type=reward_type,
                                n_eigengrasps=n_eigengrasps,
                                p_grasp_start=p_grasp_start)

    def _set_default_action(self, action: np.ndarray):
        assert action.shape == (27,)
        action = action.copy()
        move_ctrl, hand_ctrl = action[:7], action[7:]
        # Apply action to simulation.
        envs.utils.mocap_set_action(self.sim, move_ctrl * 0.05)  # Limit maximum change in position
        self.sim.data.ctrl[:] = self._act_center + hand_ctrl * self._act_range
        self.sim.data.ctrl[:] = np.clip(self.sim.data.ctrl, self._ctrl_range[:, 0],
                                        self._ctrl_range[:, 1])
