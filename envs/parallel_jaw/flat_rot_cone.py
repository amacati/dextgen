"""FlatPJRotCone environment module."""
from pathlib import Path
from typing import Optional

from gym import utils
import numpy as np

from envs.parallel_jaw.flat_base import FlatPJBase

MODEL_XML_PATH = str(Path("PJ", "flat_cone.xml"))


class FlatPJRotCone(FlatPJBase, utils.EzPickle):
    """FlatPJRotCone environment class."""

    def __init__(self, n_eigengrasps: Optional[int] = None, object_size_multiplier: float = 1.):
        """Initialize a PJ cone environment.

        Args:
            n_eigengrasps: Number of eigengrasps to use.
            object_size_multiplier: Optional multiplier to change object sizes by a fixed amount.
        """
        FlatPJBase.__init__(self,
                            object_name="cone",
                            model_xml_path=MODEL_XML_PATH,
                            object_size_multiplier=object_size_multiplier)
        utils.EzPickle.__init__(self,
                                n_eigengrasps=n_eigengrasps,
                                object_size_multiplier=object_size_multiplier)

    def _set_gripper_pose(self):
        gripper_pos = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height
                               ]) + self.gripper_init_pos  # noqa: E124
        d_pos = self.np_random.uniform(-self.gripper_init_range, self.gripper_init_range, size=2)
        gripper_pos[:2] += d_pos  # Add random initial position change
        gripper_rot = np.array([1.0, 0.0, 1.0, 0.0])
        d_rot = self.np_random.uniform(-1, 1, size=4)
        gripper_rot += (d_rot / np.linalg.norm(d_rot)) * 1  # Add random initial rotation change
        gripper_rot /= np.linalg.norm(gripper_rot)  # Renormalize for quaternion
        self.sim.data.set_mocap_pos("robot0:mocap", gripper_pos)
        self.sim.data.set_mocap_quat("robot0:mocap", gripper_rot)
