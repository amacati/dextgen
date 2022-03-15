"""UnevenPickAndPlace environment class file."""
import numpy as np
from pathlib import Path
from gym import utils
from envs.parallel_jaw.fetch import FetchEnv

MODEL_XML_PATH = str(Path("fetch", "uneven_pick_and_place.xml"))


class UnevenPickAndPlace(FetchEnv, utils.EzPickle):
    """Environment for pick and place with an uneven ground."""

    def __init__(self, reward_type: str = "sparse"):
        """Initialize the Mujoco sim."""
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        self.c_low = (1.05, 0.4, 0.4)
        self.c_high = (1.55, 1.1, 0.4)
        self.max_reset_steps = 100
        FetchEnv.__init__(
            self,
            MODEL_XML_PATH,
            has_object=True,
            block_gripper=False,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
        )
        utils.EzPickle.__init__(self, reward_type=reward_type)

    def _reset_sim(self) -> bool:
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                    -self.obj_range, self.obj_range, size=2)
            object_qpos = self.sim.data.get_joint_qpos("object0:joint")
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos("object0:joint", object_qpos)
        for _ in range(self.max_reset_steps):
            if np.linalg.norm(self.sim.data.get_site_xvelp("object0")) < 1e-3:
                if self._check_initial_pos(self.sim.data.get_site_xpos("object0")):
                    return True
                return self._reset_sim()  # Object at rest outside of bounds, retry
            self.sim.step()
        return self._reset_sim()  # Object not at rest after max_reset_steps, retry

    def _check_initial_pos(self, pos: np.ndarray) -> bool:
        return self.c_low[0] < pos[0] < self.c_high[0] and self.c_low[1] < pos[1] < self.c_high[1]
