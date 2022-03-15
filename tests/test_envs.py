import pytest
import gym
import envs  # noqa: F401


class TestEnv:

    @pytest.mark.parametrize("env", envs.available_envs)
    def test_env(self, env):
        self.run_env(env)

    @staticmethod
    def run_env(env_key):
        env = gym.make(env_key)
        env.reset()
        for _ in range(env._max_episode_steps):
            env.step(env.action_space.sample())
