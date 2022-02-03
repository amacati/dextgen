import random
import torch
import numpy as np
import gym
from mp_rl.networks import DDPGActor


def test_actor(actor, env, dev, nruns = 5):
    total_reward = 0.
    for _ in range(nruns):
        state = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                action = np.clip(actor(torch.tensor(state).to(dev)).detach().cpu().numpy(), -1, 1)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
    return total_reward/nruns


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("LunarLanderContinuous-v2")
    actor = DDPGActor(len(env.observation_space.low), len(env.action_space.low))
    for _ in range(100):
        reward = test_actor(actor, env, dev)
        if reward > 0:
            print("SUCCESS!")
            break
    print("Done")