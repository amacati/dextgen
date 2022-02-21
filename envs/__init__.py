import gym

# Register OpenAI gym environments
gym.envs.register(id='ObstacleReach-v0',
                  entry_point='envs.obstacle_reach:ObstacleReach',
                  max_episode_steps=50)

gym.envs.register(id='UnevenPickAndPlace-v0',
                  entry_point='envs.uneven_pickandplace:UnevenPickAndPlace',
                  max_episode_steps=50)
