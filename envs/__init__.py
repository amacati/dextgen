"""Init file registers environments to OpenAI's gym."""
import gym

available_envs = [
    "FlatPJCube-v0", "FlatPJSphere-v0", "FlatPJCylinder-v0", "FlatPJMesh-v0", "FlatPJAll-v0",
    "FlatSHCube-v0", "FlatSHCylinder-v0", "FlatSHSphere-v0", "FlatSHMesh-v0", "FlatSHAll-v0",
    "UnevenSHCube-v0", "ObstacleSHCube-v0", "SeaClear-v0"
]

# Register OpenAI gym environments
gym.envs.register(id="FlatPJCube-v0",
                  entry_point="envs.parallel_jaw.flat_cube:FlatPJCube",
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

gym.envs.register(id="FlatSHCube-v0",
                  entry_point="envs.shadow_hand.flat_cube:FlatSHCube",
                  max_episode_steps=50)

gym.envs.register(id="FlatSHCylinder-v0",
                  entry_point="envs.shadow_hand.flat_cylinder:FlatSHCylinder",
                  max_episode_steps=50)

gym.envs.register(id="FlatSHSphere-v0",
                  entry_point="envs.shadow_hand.flat_sphere:FlatSHSphere",
                  max_episode_steps=50)

gym.envs.register(id="FlatSHMesh-v0",
                  entry_point="envs.shadow_hand.flat_mesh:FlatSHMesh",
                  max_episode_steps=50)

gym.envs.register(id="FlatSHAll-v0",
                  entry_point="envs.shadow_hand.flat_all:FlatSHAll",
                  max_episode_steps=50)

gym.envs.register(id="UnevenSHCube-v0",
                  entry_point="envs.shadow_hand.uneven_cube:UnevenSHCube",
                  max_episode_steps=50)

gym.envs.register(id="UnevenSHMesh-v0",
                  entry_point="envs.shadow_hand.uneven_mesh:UnevenSHMesh",
                  max_episode_steps=50)

gym.envs.register(id="ObstacleSHCube-v0",
                  entry_point="envs.shadow_hand.obstacle_cube:ObstacleSHCube",
                  max_episode_steps=50)

gym.envs.register(id="SeaClear-v0",
                  entry_point="envs.seaclear.seaclear:SeaClear",
                  max_episode_steps=50)
