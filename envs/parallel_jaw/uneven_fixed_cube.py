"""UnevenPJFixedCube environment module."""
from pathlib import Path
from typing import Optional

from gym import utils
import numpy as np

from envs.parallel_jaw.uneven_base import UnevenPJBase
import envs.utils

MODEL_XML_PATH = str(Path("PJ", "uneven_cube.xml"))


class UnevenPJFixedCube(UnevenPJBase, utils.EzPickle):
    """UnevenPJFixedCube environment class."""

    def __init__(self, n_eigengrasps: Optional[int] = None, object_size_multiplier: float = 1.):
        """Initialize a PJ cube environment with uneven ground.

        Args:
            n_eigengrasps: Number of eigengrasps to use.
            object_size_multiplier: Optional multiplier to change object sizes by a fixed amount.
        """
        UnevenPJBase.__init__(self,
                              object_name="cube",
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
        pose_ctrl = np.concatenate([pos_ctrl, rot_ctrl])

        # Apply action to simulation.
        self.sim.data.ctrl[:] = self._act_center + gripper_ctrl * self._act_range
        envs.utils.mocap_set_action(self.sim, pose_ctrl)
