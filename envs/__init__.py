"""Init file registers environments to OpenAI's gym."""
import gym

available_envs = [
    "FlatPJCube-v0", "FlatPJSphere-v0", "FlatPJCylinder-v0", "FlatPJMesh-v0", "FlatPJAll-v0",
    "FlatPJOrient-v0", "FlatPJOrientEuler-v0", "FlatPJOrientQuat-v0", "FlatPJOrientAxisAngle-v0"
]

# Register OpenAI gym environments
gym.envs.register(id="FlatPJCube-v0",
                  entry_point="envs.parallel_jaw.flat_cube:FlatPJCube",
                  max_episode_steps=50)

gym.envs.register(id="FlatPJOrientQuat-v0",
                  entry_point="envs.parallel_jaw.flat_orient_quat:FlatPJOrientQuat",
                  max_episode_steps=50)

gym.envs.register(id="FlatPJOrientAxisAngle-v0",
                  entry_point="envs.parallel_jaw.flat_orient_axisangle:FlatPJOrientAxisAngle",
                  max_episode_steps=50)

gym.envs.register(id="FlatPJSphere-v0",
                  entry_point="envs.parallel_jaw.flat_sphere:FlatPJSphere",
                  max_episode_steps=50)

gym.envs.register(id="FlatPJCylinder-v0",
                  entry_point="envs.parallel_jaw.flat_cylinder:FlatPJCylinder",
                  max_episode_steps=50)

gym.envs.register(id="FlatPJMesh-v0",
                  entry_point="envs.parallel_jaw.flat_mesh:FlatPJMesh",
                  max_episode_steps=50)

gym.envs.register(id="FlatPJAll-v0",
                  entry_point="envs.parallel_jaw.flat_all:FlatPJAll",
                  max_episode_steps=50)

gym.envs.register(id="ObstacleSHCube-v0",
                  entry_point="envs.shadow_hand.obstacle_cube:ObstacleSHCube",
                  max_episode_steps=50)

gym.envs.register(id="FlatPJOrient-v0",
                  entry_point="envs.parallel_jaw.flat_orient:FlatPJOrient",
                  max_episode_steps=50)

gym.envs.register(id="FlatPJOrientEuler-v0",
                  entry_point="envs.parallel_jaw.flat_orient_euler:FlatPJOrientEuler",
                  max_episode_steps=50)
