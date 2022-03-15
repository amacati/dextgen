"""SizePickAndPlace class file."""
import numpy as np
from typing import Dict
from pathlib import Path
from gym import utils
from envs.parallel_jaw.fetch import FetchEnv
import envs.utils

MODEL_XML_PATH = str(Path("fetch", "pick_and_place.xml"))


class SizePickAndPlace(FetchEnv, utils.EzPickle):
    """Environment for cuboid grasping targets which change size on every reset."""

    def __init__(self, reward_type: str = "sparse"):
        """Initialize SizePickAndPlace environment from FetchEnv.

        Args:
            reward_type: the reward type, i.e. `sparse` or `dense`
        """
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        self.cube_low = (0.01, 0.01, 0.01)
        self.cube_high = (0.05, 0.05, 0.05)
        self._cube_size = np.zeros(3, dtype=np.float32)
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
        # Randomize object size
        self._cube_size = self.np_random.uniform(self.cube_low, self.cube_high)
        z_offset = (self._cube_size[2] - 0.025)
        self.sim.model.geom_size[-1] = self._cube_size
        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                    -self.obj_range, self.obj_range, size=2)
            object_qpos = self.sim.data.get_joint_qpos("object0:joint")
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            object_qpos[2] += z_offset
            self.sim.data.set_joint_qpos("object0:joint", object_qpos)
        self.sim.forward()
        return True

    def _get_obs(self) -> Dict[str, np.ndarray]:
        # positions
        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip") * dt
        robot_qpos, robot_qvel = envs.utils.robot_get_obs(self.sim)
        object_pos = self.sim.data.get_site_xpos("object0")
        # rotations
        object_rot = envs.rotations.mat2euler(self.sim.data.get_site_xmat("object0"))
        # velocities
        object_velp = self.sim.data.get_site_xvelp("object0") * dt
        object_velr = self.sim.data.get_site_xvelr("object0") * dt
        # gripper state
        object_rel_pos = object_pos - grip_pos
        object_velp -= grip_velp
        gripper_state = robot_qpos[-2:]
        gripper_vel = (robot_qvel[-2:] * dt)  # change to a scalar if the gripper is made symmetric

        achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate([
            grip_pos,
            self._cube_size,
            object_pos.ravel(),
            object_rel_pos.ravel(),
            gripper_state,
            object_rot.ravel(),
            object_velp.ravel(),
            object_velr.ravel(),
            grip_velp,
            gripper_vel,
        ])

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }
