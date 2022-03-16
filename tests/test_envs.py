import pytest
import gym
import envs  # noqa: F401


@pytest.mark.parametrize("env", envs.available_envs)
def test_env(env):
    run_env(env)


def run_env(env_key):
    env = gym.make(env_key)
    env.reset()
    for _ in range(env._max_episode_steps):
        env.step(env.action_space.sample())
