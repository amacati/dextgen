"""ShadowHandGravity class file."""
from typing import Optional

from gym import utils
import numpy as np

from envs.shadow_hand.shadowhand_base import ShadowHandBase


class ShadowHandGravity(ShadowHandBase, utils.EzPickle):
    """Environment for pick and place with the ShadowHand and eigengrasps."""

    def __init__(self,
                 reward_type: str = "sparse",
                 n_eigengrasps: Optional[int] = None,
                 n_increase_epochs: int = 100):
        """Initialize the Mujoco sim.

        Params:
            reward_type: Choice of reward formular.
            n_eigengrasps: Number of eigengrasp vectors the agent gets as action input.
            n_increase_epochs: Number of epochs over which gravity increases to 1.
        """
        self.n_increase_epochs = n_increase_epochs
        n_actions = 3 + (n_eigengrasps or 20)
        super().__init__(n_actions=n_actions,
                         reward_type=reward_type,
                         p_grasp_start=0,
                         n_eigengrasps=n_eigengrasps)
        utils.EzPickle.__init__(self,
                                reward_type=reward_type,
                                n_eigengrasps=n_eigengrasps,
                                n_increase_epochs=n_increase_epochs)
        self.sim.model.opt.gravity[-1] = 0

    def epoch_callback(self, epoch: int):
        """Increase environment gravity stepwise after each epoch.

        Args:
            epoch: Current training epoch.
            max_epoch: Maximum number of epochs.
        """
        self.sim.model.opt.gravity[-1] = max(min(1, epoch / (self.n_increase_epochs)), 0) * -9.81

    def _reset_sim(self) -> bool:
        self.sim.set_state(self.initial_state)
        # Randomize start position of object.
        object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
            -self.obj_range, self.obj_range, size=2)
        while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
            object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                -self.obj_range, self.obj_range, size=2)
        object_qpos = self.sim.data.get_joint_qpos("object0:joint")
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        # Stop random placement in the air when gravity approaches real value
        if np.random.rand() > 0.5 and self.sim.model.opt.gravity[-1] < 5:
            object_qpos[2] += np.random.rand() * 0.2
        self.sim.data.set_joint_qpos("object0:joint", object_qpos)
        self.sim.forward()
        return True
