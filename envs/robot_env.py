"""Robot_env copy from the robot environments in OpenAI's gym.

Minor changes have been made for our use case.

See https://github.com/Farama-Foundation/Gym-Robotics.
"""

import os
import copy
import numpy as np
from typing import Optional, Union, Dict, Tuple, Any, List

import gym
from gym import error, spaces
from gym.utils import seeding

try:
    import mujoco_py as mjpy
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: \
            https://github.com/openai/mujoco-py/.)".format(e))

DEFAULT_SIZE = 500


class RobotEnv(gym.GoalEnv):
    """Superclass for robot environments."""

    def __init__(self, model_path: str, initial_qpos: dict, n_actions: int, n_substeps: int):
        """Load assets, models and start the Mujoco sim.

        Args:
            model_path: Path to the sim description xml file.
            initial_qpos: Initial pose information for objects in the sim.
            n_actions: Action dimension.
            n_substeps: Number of internal Mujoco steps for one single gym step.
        """
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not os.path.exists(fullpath):
            raise IOError("File {} does not exist".format(fullpath))

        model = mjpy.load_model_from_path(fullpath)
        self.sim = mjpy.MjSim(model, nsubsteps=n_substeps)
        self.viewer = None
        self._viewers = {}
        self._contact_info = False

        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": int(np.round(1.0 / self.dt)),
        }

        self.seed(1)
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())

        self.goal = self._sample_goal()
        obs = self._get_obs()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_actions,), dtype="float32")
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(-np.inf,
                                        np.inf,
                                        shape=obs["achieved_goal"].shape,
                                        dtype="float32"),
                achieved_goal=spaces.Box(-np.inf,
                                         np.inf,
                                         shape=obs["achieved_goal"].shape,
                                         dtype="float32"),
                observation=spaces.Box(-np.inf,
                                       np.inf,
                                       shape=obs["observation"].shape,
                                       dtype="float32"),
            ))
        self.seed()

    @property
    def dt(self) -> float:
        """Time delta property."""
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    # Env methods
    # ----------------------------

    def seed(self, seed: Optional[float] = None) -> List[float]:
        """Set the random seed for the environment.

        Args:
            seed: The random seed.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: np.ndarray) -> Tuple[dict, float, bool, Any]:
        """Take a step in the environment with the given action.

        Args:
            action: Agent action.
        """
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        reward = self.compute_reward(obs["achieved_goal"], self.goal, None)
        info = {
            "is_success": reward == 0,
        }
        if self._contact_info:
            info["contact_info"] = self._get_contact_info()

        return obs, reward, done, info

    def reset(self) -> Dict[str, np.ndarray]:
        """Attempt to reset the simulator.

        Since we randomize initial conditions, it is possible to get into a state with numerical
        issues (e.g. due to penetration or Gimbel lock) or we may not achieve an initial condition
        (e.g. an object is within the hand). In this case, we just keep randomizing until we
        eventually achieve a valid initial configuration.
        """
        super(RobotEnv, self).reset()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        return obs

    def close(self):
        """Close the viewer."""
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def render(self,
               mode: str = "human",
               width: int = DEFAULT_SIZE,
               height: int = DEFAULT_SIZE) -> Optional[np.ndarray]:
        """Render the current sim state.

        Args:
            mode: Render mode.
            width: Render window width.
            heights: Render window height.
        """
        self._render_callback()
        if mode == "rgb_array":
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == "human":
            self._get_viewer(mode).render()

    def enable_contact_info(self, val: bool = True):
        """Enable contact information between the gripper and the object in the info step return.

        Has to be a function since gym wraps the environment in a `TimeLimit` object which does not
        forward attribute changes.

        Args:
            val: Flag to enable or disable contact information. Default is True.
        """
        self._contact_info = val

    def _get_viewer(self, mode: str) -> Union[mjpy.MjViewer, mjpy.MjRenderContextOffscreen]:
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = mjpy.MjViewer(self.sim)
            elif mode == "rgb_array":
                self.viewer = mjpy.MjRenderContextOffscreen(self.sim, device_id=-1)
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    # Extension methods
    # ----------------------------

    def _reset_sim(self) -> bool:
        """Reset a simulation and indicate whether or not it was successful.

        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        return True

    def _get_obs(self):
        """Return the observation."""
        raise NotImplementedError()

    def _set_action(self, action: np.ndarray):
        """Apply the given action to the simulation."""
        raise NotImplementedError()

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray):
        """Indicate whether or not the achieved goal successfully achieved the desired goal."""
        raise NotImplementedError()

    def _sample_goal(self):
        """Sample a new goal and return it."""
        raise NotImplementedError()

    def _env_setup(self, initial_qpos: dict):
        """Configure the environment.

        Can be used to configure initial state and extract information from the simulation.
        """
        pass

    def _viewer_setup(self):
        """Configure the viewer, e.g. set the camera position."""
        pass

    def _render_callback(self):
        """Execute callback before rendering.

        Can be used to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """Execute callback after stepping once in the simulation.

        Can be used to enforce additional constraints on the simulation state.
        """
        pass
