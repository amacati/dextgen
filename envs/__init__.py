"""Init file registers environments to OpenAI's gym."""
import gym

available_envs = [
    'ObstacleReach-v0', 'UnevenPickAndPlace-v0', 'SeaClearPickAndPlace-v0', 'SizePickAndPlace-v0',
    'OrientPickAndPlace-v0', 'ShadowHandPickAndPlace-v0', 'ShadowHandEigengrasps-v0',
    'ShadowHandGravity-v0'
]

# Register OpenAI gym environments
gym.envs.register(id='ObstacleReach-v0',
                  entry_point='envs.parallel_jaw.obstacle_reach:ObstacleReach',
                  max_episode_steps=50)

gym.envs.register(id='UnevenPickAndPlace-v0',
                  entry_point='envs.parallel_jaw.uneven_pickandplace:UnevenPickAndPlace',
                  max_episode_steps=50)

gym.envs.register(id='SeaClearPickAndPlace-v0',
                  entry_point='envs.parallel_jaw.seaclear_pickandplace:SeaClearPickAndPlace',
                  max_episode_steps=50)

gym.envs.register(id='SizePickAndPlace-v0',
                  entry_point='envs.parallel_jaw.size_pickandplace:SizePickAndPlace',
                  max_episode_steps=50)

gym.envs.register(id='OrientPickAndPlace-v0',
                  entry_point='envs.parallel_jaw.orient_pickandplace:OrientPickAndPlace',
                  max_episode_steps=50)

gym.envs.register(id='ShadowHandPickAndPlace-v0',
                  entry_point='envs.shadow_hand.shadowhand_pickandplace:ShadowHandPickAndPlace',
                  max_episode_steps=50)

gym.envs.register(id='ShadowHandEigengrasps-v0',
                  entry_point='envs.shadow_hand.shadowhand_eigen:ShadowHandEigengrasps',
                  max_episode_steps=50)

gym.envs.register(id='ShadowHandGravity-v0',
                  entry_point='envs.shadow_hand.shadowhand_gravity:ShadowHandGravity',
                  max_episode_steps=50)
