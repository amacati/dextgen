import gym
from mp_rl.core.ddpg import DDPG
from envs.forbidden_forest import ForbiddenForest

if __name__ == "__main__":
    gym.envs.register(id='ForbiddenForest-v0',
                      entry_point='envs.forbidden_forest:ForbiddenForest',
                      max_episode_steps=50)
    gym.make("ForbiddenForest-v0")
