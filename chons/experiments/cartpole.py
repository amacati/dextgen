import logging
from queue import Queue
import multiprocessing
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
from tqdm import tqdm
import yaml
# Allow script to execute as main without import errors. Yes I know this is hacky
if __name__ == "__main__":
    from memory import MemoryBuffer
    from utils import running_average
else:
    from .memory import MemoryBuffer
    from .utils import running_average


logger = logging.getLogger(__name__)


# DQNetwork
class DQN(nn.Module):
    
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.l1 = nn.Linear(input_size, 64)
        self.l1_afct = nn.ReLU()
        self.l2 = nn.Linear(64, 32)
        self.l2_afct = nn.ReLU()
        self.l3 = nn.Linear(32, output_size)
    
    def forward(self, x: torch.Tensor):
        _x = self.l1_afct(self.l1(x))
        _x = self.l2_afct(self.l2(_x))
        return self.l3(_x)


def cartpole(config, result: Queue):
    # Check if running as main process to toggle status bar
    mp_flag = multiprocessing.parent_process() is not None
    logger.debug(f"Running in {('MP' if mp_flag else 'SP')} mode.")
    logger.debug(f"Config: {config}")
    # Setup constants, hyperparameters, bookkeeping
    env = gym.make("CartPole-v1")
    n_episodes = config["n_episodes"]
    n_states = len(env.observation_space.low)
    n_actions = env.action_space.n
    gamma = config["gamma"]
    lr = config["lr"]
    eps_max = config["eps_max"]
    eps_min = config["eps_min"]
    dqn = DQN(n_states, n_actions)
    optimizer = torch.optim.Adam(dqn.parameters(), lr=lr)
    buffer = MemoryBuffer(config["buffer_size"])
    batch_size = config["batch_size"]
    episode_reward_list = []

    save_policy = config["save_policy"]

    # Training setup
    logger.debug("Training start")
    if not mp_flag:
        status_bar = tqdm(total=n_episodes, desc="Training iterations", position=0)
        reward_log = tqdm(total=0, position=1, bar_format='{desc}')
    for i in range(n_episodes):
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            if np.random.rand() < eps_max - (eps_max - eps_min)*i/n_episodes:  # Linear decay exploration
                action = np.random.randint(0, n_actions)
            else:
                qvalues = dqn(torch.tensor(state, requires_grad=False))
                action = torch.argmax(qvalues).item()
            next_state, reward, done, _ = env.step(action)
            buffer.append((state, action, reward, next_state, done))
            ep_reward += reward
            state = next_state
            if len(buffer)>=batch_size:
                optimizer.zero_grad()
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                qcurr = dqn(torch.tensor(states))
                rcurr = qcurr[range(batch_size), actions]
                with torch.no_grad():
                    rnext = torch.max(dqn(torch.tensor(next_states, requires_grad=False)), 1)[0]
                rewards = torch.tensor(rewards) + rnext * (1 - torch.tensor(dones)) * gamma
                loss = nn.functional.mse_loss(rcurr, rewards)
                loss.backward()
                nn.utils.clip_grad_norm_(dqn.parameters(), 1.)
                optimizer.step()
        episode_reward_list.append(ep_reward)
        av_reward = running_average(episode_reward_list)[-1]
        if not mp_flag:
            reward_log.set_description_str("Current running average reward: {:.1f}".format(av_reward))
            status_bar.update()
        if av_reward > 200:
            logger.debug("Training stopped, agent has solved the task.")
            break
    env.close()
    logger.debug("Training has ended")

    if save_policy:
        path = Path(__file__).resolve().parent / "cartpole_dqn.pt"
        logger.debug(f"Saving model to {path}")
        torch.save(dqn.state_dict(), path)
        logger.debug(f"Saving complete")
    
    result.put(episode_reward_list)
    return


if __name__ == "__main__":
    logging.basicConfig()
    logger.setLevel(logging.INFO)
    logger.debug("Loading config")
    path = Path(__file__).resolve().parents[1] / "config" / "experiment_config.yaml"
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    logger.debug("Loading successful, starting experiment")
    result = Queue()
    cartpole(config["cartpole"], result)
    logger.debug("Training finished")
    # Visualize reward
    episode_reward_list = result.get()
    fig, ax = plt.subplots()
    ax.plot(episode_reward_list)
    smooth_reward = running_average(episode_reward_list)
    index = range(len(episode_reward_list)-len(smooth_reward), len(episode_reward_list))
    ax.plot(index, smooth_reward)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Accumulated reward')
    ax.set_title('Agent performance over time')
    ax.legend(["Episode reward", "Running average reward"])
    plt.show()
