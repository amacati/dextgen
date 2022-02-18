"""Project setup file."""

from setuptools import setup

setup(
    name="mp_rl",
    version="0.1",
    author="Martin Schuck",
    author_email="martin.schuck@tum.de",
    description=("A multiprocessing reinforcement learning module for grasping."),
    keywords="RL MP DDPG DDP",
    packages=['mp_rl', 'tests', 'envs'],
)
