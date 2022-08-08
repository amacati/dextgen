"""FlatPJFixedCone environment module."""
from pathlib import Path
from typing import Optional

from gym import utils
import numpy as np

from envs.parallel_jaw.flat_base import FlatPJBase
import envs.utils

MODEL_XML_PATH = str(Path("PJ", "flat_cone.xml"))


class FlatPJFixedCone(FlatPJBase, utils.EzPickle):
    """FlatPJFixedCone environment class."""

    def __init__(self, n_eigengrasps: Optional[int] = None, object_size_multiplier: float = 1.):
        """Initialize a PJ cone environment.

        Args:
            n_eigengrasps: Number of eigengrasps to use.
            object_size_multiplier: Optional multiplier to change object sizes by a fixed amount.
        """
        FlatPJBase.__init__(self,
                            object_name="cone",
                            model_xml_path=MODEL_XML_PATH,
                            object_size_multiplier=object_size_multiplier,
                            n_actions=4)
        utils.EzPickle.__init__(self,
                                n_eigengrasps=n_eigengrasps,
                                object_size_multiplier=object_size_multiplier)

    def _set_action(self, action: np.ndarray):
        assert action.shape == (self.n_actions,)
        action = (action.copy())  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = np.array([1., 0., 1., 0.])
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        envs.utils.ctrl_set_action(self.sim, action)
        envs.utils.mocap_set_action(self.sim, action)
