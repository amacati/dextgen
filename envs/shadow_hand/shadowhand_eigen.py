"""ShadowHandEigengrasps class file."""
from gym import utils

from envs.shadow_hand.shadowhand_base import ShadowHandBase


class ShadowHandEigengrasps(ShadowHandBase, utils.EzPickle):
    """Environment for pick and place with the ShadowHand and eigengrasps."""

    def __init__(self,
                 reward_type: str = "sparse",
                 n_eigengrasps: int = 1,
                 p_grasp_start: float = 0.):
        """Initialize the Mujoco sim.

        Params:
            reward_type: Choice of reward formular.
            n_eigengrasps: Number of eigengrasp vectors the agent gets as action input.
            p_grasp_start: Fraction of episode starts with pregrasped objects.
        """
        n_actions = 3 + n_eigengrasps
        super().__init__(n_actions=n_actions,
                         reward_type=reward_type,
                         p_grasp_start=p_grasp_start,
                         n_eigengrasps=n_eigengrasps)
        utils.EzPickle.__init__(self,
                                reward_type=reward_type,
                                n_eigengrasps=n_eigengrasps,
                                p_grasp_start=p_grasp_start)
