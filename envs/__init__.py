"""Init file registers environments to OpenAI's gym."""
import gym

available_envs = ["FlatPJCube-v0", "FlatPJOrientQuat-v0"]

# Register OpenAI gym environments
gym.envs.register(id="FlatPJCube-v0",
                  entry_point="envs.parallel_jaw.flat_cube:FlatPJCube",
                  max_episode_steps=50)

gym.envs.register(id="FlatPJOrientQuat-v0",
                  entry_point="envs.parallel_jaw.flat_orient_quat:FlatPJOrientQuat",
                  max_episode_steps=50)
