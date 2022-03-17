import random

import pytest
import gym
import envs  # noqa: F401


@pytest.mark.parametrize("env", envs.available_envs)
def test_env(env):
    env = gym.make(env)
    env.reset()
    for _ in range(env._max_episode_steps):
        env.step(env.action_space.sample())


@pytest.mark.parametrize("env", [e for e in envs.available_envs if "ShadowHand" in e])
def test_shadowhand_eigen(env):
    if "PickAndPlace" in env:
        return
    env = gym.make(env, n_eigengrasps=random.randint(1, 20))
    env.reset()
    for _ in range(env._max_episode_steps):
        env.step(env.action_space.sample())
