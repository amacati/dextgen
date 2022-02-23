import gym
import envs  # noqa: F401


class TestEnv:

    def test_obstacle_reach_env(self):
        self.run_env(env_key="ObstacleReach-v0")

    def test_uneven_pickandplace_env(self):
        self.run_env(env_key="UnevenPickAndPlace-v0")

    def test_seaclear_pickandplace_env(self):
        self.run_env(env_key="SeaClearPickAndPlace-v0")

    def test_size_pickandplace_env(self):
        self.run_env(env_key="SizePickAndPlace-v0")

    def test_shadowhand_pickandplace_env(self):
        self.run_env(env_key="ShadowHandPickAndPlace-v0")

    @staticmethod
    def run_env(env_key):
        env = gym.make(env_key)
        env.reset()
        for _ in range(env._max_episode_steps):
            env.step(env.action_space.sample())
