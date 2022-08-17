"""Project setup file."""
from setuptools import setup, find_packages

setup(name="dextgen_package_collection",
      version="0.1",
      author="Martin Schuck",
      author_email="martin.schuck@tum.de",
      description=("A collection of reinforcement learning modules for grasping."),
      keywords="RL MP DDPG DDP",
      packages=find_packages(),
      include_package_data=True)
