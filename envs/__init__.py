import gym

# Register OpenAI gym
gym.envs.register(id='ObstacleReach-v0',
                  entry_point='envs.obstacle_reach:ObstacleReach',
                  max_episode_steps=50)
