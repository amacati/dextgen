import argparse
import logging
from pathlib import Path
import time

import torch
import gym

from mp_rl.ddpg import load_ddpg


def test_lunar():
    path = Path(__file__).parent/"mp_rl"/"lunar_lander"/"ddpg.pkl"
    ddpg = load_ddpg(path)
    env = gym.make("LunarLanderContinuous-v2")
    state = env.reset()
    t_reward = 0
    done = False
    while not done:
        action = ddpg.action(torch.unsqueeze(torch.as_tensor(state), 0))[0]
        next_state, reward, done, _ = env.step(action)
        t_reward += reward
        state = next_state
        env.render()
        time.sleep(0.01)
    logger.info(f"Episode reward: {reward}")
    
def test_fetch():
    raise NotImplementedError

def test_fetch_her():
    raise NotImplementedError

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env", help="Selects the gym environment", choices=["lunar", "fetch", "fetch_her"])
    parser.add_argument('--loglvl', help="Logger levels", choices=["DEBUG", "INFO", "WARN", "ERROR"],
                        default="INFO")
    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    loglvls = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARN": logging.WARN, "ERROR": logging.ERROR}
    logging.basicConfig()
    logging.getLogger().setLevel(loglvls[args.loglvl])
    env_tests = {"lunar": test_lunar, "fetch": test_fetch, "fetch_her": test_fetch_her}
    env_tests[args.env]()