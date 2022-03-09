"""Test a previously trained agent on an OpenAI gym environment."""

import argparse
import logging
from pathlib import Path
import time
from typing import Optional, Tuple

import pickle
import torch
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import mujoco_py
import envs  # Import registers environments with gym  # noqa: F401

from mp_rl.core.utils import unwrap_obs
from mp_rl.core.actor import ActorNetwork


class MujocoVideoRecorder(VideoRecorder):

    def __init__(self, *args, resolution: Optional[Tuple[int]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.resolution = resolution

    def capture_frame(self):
        if self.resolution is None:
            return super().capture_frame()
        if not self.functional or self._closed:
            return
        frame = self.env.render("rgb_array", width=self.resolution[0], height=self.resolution[1])
        if frame is None:
            if self._async:
                return
            self.broken = True
        else:
            self.last_frame = frame
            self._encode_image_frame(frame)


def parse_args() -> argparse.Namespace:
    """Parse arguments for the gym environment and logging levels.

    Returns:
        The parsed arguments as a namespace.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",
                        help="Selects the gym environment",
                        choices=[
                            "FetchReach-v1", "FetchPickAndPlace-v1", "ObstacleReach-v0",
                            "UnevenPickAndPlace-v0", "SeaClearPickAndPlace-v0",
                            "SizePickAndPlace-v0", "ShadowHandPickAndPlace-v0",
                            "OrientPickAndPlace-v0", "ShadowHandEigengrasps-v0"
                        ],
                        default="FetchReach-v1")
    parser.add_argument("--loglvl",
                        help="Logger levels",
                        choices=["DEBUG", "INFO", "WARN", "ERROR"],
                        default="INFO")
    parser.add_argument("--render", help="Render flag", choices=["y", "n"], default="y")
    parser.add_argument("--record", help="Record video flag", choices=["y", "n"], default="n")
    parser.add_argument("--ntests", help="Number of evaluation runs", default=10, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    logger = logging.getLogger("GymTestScript")
    loglvls = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARN": logging.WARN,
        "ERROR": logging.ERROR
    }
    logging.basicConfig()
    logging.getLogger().setLevel(loglvls[args.loglvl])
    if hasattr(args, "kwargs") and args.kwargs:
        env = gym.make(args.env, **args.kwargs)
    else:
        env = gym.make(args.env)
    size_s = len(env.observation_space["observation"].low) + len(
        env.observation_space["desired_goal"].low)
    size_a = len(env.action_space.low)
    actor = ActorNetwork(size_s, size_a)
    path = Path(__file__).parent / "saves" / args.env
    actor.load_state_dict(torch.load(path / "actor.pt"))
    with open(path / "state_norm.pkl", "rb") as f:
        state_norm = pickle.load(f)
    with open(path / "goal_norm.pkl", "rb") as f:
        goal_norm = pickle.load(f)
    success = 0.
    render = args.render == "y"
    record = args.record == "y"
    if record:
        path = Path(__file__).parent / "video" / (args.env + ".mp4")
        recorder = MujocoVideoRecorder(env, path=str(path), resolution=(1920, 1080))
        logger.info("Recording video, environment rendering disabled")
    early_stop = 0
    for _ in range(args.ntests):
        state, goal, _ = unwrap_obs(env.reset())
        done = False
        while not done:
            state, goal = state_norm(state), goal_norm(goal)
            state, goal = torch.as_tensor(state,
                                          dtype=torch.float32), torch.as_tensor(goal,
                                                                                dtype=torch.float32)
            with torch.no_grad():
                action = actor(torch.cat([state, goal]))
            next_obs, reward, done, info = env.step(action.numpy())
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            state, goal, _ = unwrap_obs(next_obs)
            early_stop = (early_stop + 1) if not reward else 0
            if record:
                recorder.capture_frame()
            if render and not record:
                try:
                    env.render()
                    time.sleep(0.04)  # Gym operates on 25 Hz
                except mujoco_py.cymj.GlfwError:
                    logger.warning("No display available, rendering disabled")
                    render = False
            if early_stop == 5 and not record:
                break
        success += info["is_success"]
    if record:
        recorder.close()
        (Path(__file__).parent / "video" / (args.env + ".meta.json")).unlink()  # Delete metafile
    logger.info(f"Agent success rate: {success/args.ntests:.2f}")
