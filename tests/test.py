"""Small test script for miscellaneous stuff."""
import gym
import envs
import numpy as np
import time

env = gym.make("ObstacleReach-v0")
obs = env.reset()
print(env.observation_space)
while True:
    # env.step(env.action_space.sample())
    env.render()
