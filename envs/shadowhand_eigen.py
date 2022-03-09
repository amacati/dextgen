"""ShadowHandPickAndPlace class file."""
from pathlib import Path
from typing import Dict
from gym import utils
import numpy as np
import json
import envs.robot_env
import envs.utils
import envs.rotations

MODEL_XML_PATH = str(Path("shfetch", "shadowhand_pick_and_place.xml"))

# The eigengrasps are exctracted from joint configurations obtained by fitting the ShadowHand to
# hand poses from the ContactPose dataset. For more information, see
# https://github.com/amacati/sh_eigen  TODO: Make repository public
with open(Path(__file__).parent / "eigengrasps.json", "r") as f:
    eigengrasps = json.load(f)
    assert all([len(value["joints"]) == 20 for value in eigengrasps.values()])
EIGENGRASPS = np.array([eigengrasps[str(i)]["joints"] for i in range(len(eigengrasps))])

DEFAULT_INITIAL_QPOS = {
    "robot0:slide0": 0.4049,  # Robot arm
    "robot0:slide1": 0.48,
    "robot0:slide2": 0.0,
    "robot0:WRJ1": -0.16514339750464327,  # ShadowHand
    "robot0:WRJ0": -0.31973286565062153,
    "robot0:FFJ3": 0.14340512546557435,
    "robot0:FFJ2": 0.32028208333591573,
    "robot0:FFJ1": 0.7126053607727917,
    "robot0:FFJ0": 0.6705281001412586,
    "robot0:MFJ3": 0.000246444303701037,
    "robot0:MFJ2": 0.3152655251085491,
    "robot0:MFJ1": 0.7659800313729842,
    "robot0:MFJ0": 0.7323156897425923,
    "robot0:RFJ3": 0.00038520700007378114,
    "robot0:RFJ2": 0.36743546201985233,
    "robot0:RFJ1": 0.7119514095008576,
    "robot0:RFJ0": 0.6699446327514138,
    "robot0:LFJ4": 0.0525442258033891,
    "robot0:LFJ3": -0.13615534724474673,
    "robot0:LFJ2": 0.39872030433433003,
    "robot0:LFJ1": 0.7415570009679252,
    "robot0:LFJ0": 0.704096378652974,
    "robot0:THJ4": 0.003673823825070126,
    "robot0:THJ3": 0.5506291436028695,
    "robot0:THJ2": -0.014515151997119306,
    "robot0:THJ1": -0.0015229223564485414,
    "robot0:THJ0": -0.7894883021600622,
    "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0]  # Object default location
}


class ShadowHandEigengrasps(envs.robot_env.RobotEnv, utils.EzPickle):
    """Environment for pick and place with the ShadowHand and eigengrasps."""
    EIGENGRASPS = EIGENGRASPS

    def __init__(self, reward_type: str = "sparse", n_eigengrasps: int = 1):
        """Initialize the Mujoco sim."""
        self.c_low = (1.05, 0.4, 0.4)
        self.c_high = (1.55, 1.1, 0.4)
        self.max_reset_steps = 100
        self.distance_threshold = 0.05
        self.target_in_the_air = True
        self.target_range = 0.15
        self.target_offset = 0.0
        self.gripper_extra_height = 0.30
        self.reward_type = reward_type
        self.obj_range = 0.15
        assert n_eigengrasps <= 20
        self.n_eigengrasps = n_eigengrasps
        super().__init__(model_path=MODEL_XML_PATH,
                         n_substeps=20,
                         n_actions=3 + n_eigengrasps,
                         initial_qpos=DEFAULT_INITIAL_QPOS)
        utils.EzPickle.__init__(self, reward_type=reward_type)
        self._ctrl_range = self.sim.model.actuator_ctrlrange
        self._act_range = (self._ctrl_range[:, 1] - self._ctrl_range[:, 0]) / 2.0
        self._act_center = (self._ctrl_range[:, 1] + self._ctrl_range[:, 0]) / 2.0

    def _sample_goal(self) -> np.ndarray:
        goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
            -self.target_range, self.target_range, size=3)
        goal += self.target_offset
        goal[2] = self.height_offset
        if self.target_in_the_air and self.np_random.uniform() < 0.5:
            goal[2] += self.np_random.uniform(0.1, 0.45)
        return goal.copy()

    def _get_obs(self) -> Dict[str, np.ndarray]:
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
        hand_state = robot_qpos[-24:]

        achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate([
            grip_pos,
            grip_velp,
            hand_state,
            object_pos.ravel(),
            object_rel_pos.ravel(),
            object_rot.ravel(),
            object_velp.ravel(),
            object_velr.ravel(),
        ])
        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy()
        }

    def _set_action(self, action: np.ndarray):
        assert action.shape == (3 + self.n_eigengrasps,)
        action = (action.copy())  # ensure that we don't change the action outside of this scope
        pos_ctrl, hand_ctrl = action[:3], action[3:]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1.0, 0.0, 1.0, 0.0]  # fixed rotation of the end effector as a quaternion
        action = np.concatenate([pos_ctrl, rot_ctrl])

        # Transform hand controls to eigengrasps
        hand_ctrl[:] = 1
        hand_ctrl = envs.utils.map_sh2mujoco(hand_ctrl @ self.EIGENGRASPS[:self.n_eigengrasps])
        np.clip(hand_ctrl, -1, 1, out=hand_ctrl)

        # Apply action to simulation.
        envs.utils.mocap_set_action(self.sim, action)
        self.sim.data.ctrl[:] = self._act_center + hand_ctrl * self._act_range
        self.sim.data.ctrl[:] = np.clip(self.sim.data.ctrl, self._ctrl_range[:, 0],
                                        self._ctrl_range[:, 1])

    def _is_success(self, achieved_goal: np.ndarray, goal: np.ndarray) -> np.ndarray:
        d = envs.utils.goal_distance(achieved_goal, goal)
        return (d < self.distance_threshold).astype(np.float32)

    def compute_reward(self, achieved_goal: np.ndarray, goal: np.ndarray, _) -> np.ndarray:
        """Compute the agent reward for the achieved goal.

        Args:
            achieved_goal: Achieved goal.
            goal: Desired goal.
        """
        # Compute distance between goal and the achieved goal.
        d = envs.utils.goal_distance(achieved_goal, goal)
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def _env_setup(self, initial_qpos: np.ndarray):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        envs.utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.52, 0.005, -0.431 + self.gripper_extra_height
                                  ]) + self.sim.data.get_site_xpos("robot0:grip")  # noqa: E124
        gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        self.sim.data.set_mocap_pos("robot0:mocap", gripper_target)
        self.sim.data.set_mocap_quat("robot0:mocap", gripper_rotation)
        for _ in range(10):
            self.sim.step()

        self.initial_gripper_xpos = self.sim.data.get_site_xpos("robot0:grip").copy()
        self.height_offset = self.sim.data.get_site_xpos("object0")[2]

    def _reset_sim(self) -> bool:
        self.sim.set_state(self.initial_state)
        # Randomize start position of object.
        object_xpos = self.initial_gripper_xpos[:2]
        while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
            object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                -self.obj_range, self.obj_range, size=2)
        object_qpos = self.sim.data.get_joint_qpos("object0:joint")
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        self.sim.data.set_joint_qpos("object0:joint", object_qpos)
        self.sim.forward()
        return True

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id("target0")
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()
