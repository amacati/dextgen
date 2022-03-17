"""ShadowHandPickAndPlace class file."""
from gym import utils

from envs.shadow_hand.shadowhand_base import ShadowHandBase


class ShadowHandPickAndPlace(ShadowHandBase, utils.EzPickle):
    """Environment for pick and place with the ShadowHand."""

    def __init__(self, reward_type: str = "sparse", p_grasp_start: float = 0.):
        """Initialize the Mujoco sim.

        Params:
            reward_type: Choice of reward formular.
            p_grasp_start: Fraction of episode starts with pregrasped objects.
        """
        super().__init__(n_actions=23, reward_type=reward_type, p_grasp_start=p_grasp_start)
        utils.EzPickle.__init__(self, reward_type=reward_type, p_grasp_start=p_grasp_start)
