"""Init file registers environments to OpenAI's gym."""
import gym

# Register OpenAI gym environments
gym.envs.register(id='ObstacleReach-v0',
                  entry_point='envs.obstacle_reach:ObstacleReach',
                  max_episode_steps=50)

gym.envs.register(id='UnevenPickAndPlace-v0',
                  entry_point='envs.uneven_pickandplace:UnevenPickAndPlace',
                  max_episode_steps=50)

gym.envs.register(id='SeaClearPickAndPlace-v0',
                  entry_point='envs.seaclear_pickandplace:SeaClearPickAndPlace',
                  max_episode_steps=50)

gym.envs.register(id='SizePickAndPlace-v0',
                  entry_point='envs.size_pickandplace:SizePickAndPlace',
                  max_episode_steps=50)
