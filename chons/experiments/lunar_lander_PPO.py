import logging
import time
import yaml
import multiprocessing
from queue import Queue
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
from tqdm import tqdm
from memory import TrajectoryBuffer
from utils import running_average


logger = logging.getLogger(__name__)


def gaussian_pdf(values: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:    
    return torch.prod(1/torch.sqrt(2*np.pi*var)*torch.exp(-(values - mean)**2/(2*var)), -1)


def ppo_clip_loss(pi_theta: torch.Tensor, pi_theta_old: torch.Tensor, advantage: torch.Tensor, 
                  epsilon: float) -> torch.Tensor:
    importance = pi_theta/pi_theta_old
    return -torch.mean(torch.minimum(importance*advantage, clip(importance, epsilon)*advantage))


def clip(x: torch.Tensor, epsilon: float) -> torch.Tensor:
    return torch.maximum(torch.ones_like(x) - epsilon, torch.minimum(x, torch.ones_like(x) + epsilon))

# PPO Actor
class PPOActor(nn.Module):
    
    def __init__(self, n_states: int, n_actions: int):
        super().__init__()
        self.l1 = nn.Linear(n_states, 400)
        self.l1_mh = nn.Linear(400, 200)
        self.l2_mh = nn.Linear(200, n_actions)
        self.l1_varh = nn.Linear(400, 200)
        self.l2_varh = nn.Linear(200, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.l1(x))
        x_mean = torch.relu(self.l1_mh(x))
        x_mean = torch.tanh(self.l2_mh(x_mean))
        x_var = torch.relu(self.l1_varh(x))
        x_var = torch.sigmoid(self.l2_varh(x_var))
        return x_mean, x_var
    
# PPO Critic
class PPOCritic(nn.Module):
    
    def __init__(self, n_states: int):
        super().__init__()
        self.l1 = nn.Linear(n_states, 400)
        self.l2 = nn.Linear(400, 200)
        self.l3 = nn.Linear(200, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return self.l3(x)
    

def lunarlander_ppo(config: dict, result: Queue):
    # Check if running as main process to toggle status bar
    mp_flag = multiprocessing.parent_process() is not None
    logger.debug(f"Running in {('MP' if mp_flag else 'SP')} mode.")
    logger.debug(f"Config: {config}")
    # Setup constants, hyperparameters, bookkeeping
    env = gym.make("LunarLanderContinuous-v2")
    n_episodes = config["n_episodes"]
    n_states = len(env.observation_space.low)
    n_actions = len(env.action_space.low)
    logger.debug(env.action_space)
    gamma = config["gamma"]
    actor_lr = config["actor_lr"]
    critic_lr = config["critic_lr"]
    epsilon = config["epsilon"]
    train_iterations = config["train_iterations"]
    
    actor = PPOActor(n_states, n_actions)
    critic = PPOCritic(n_states)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    buffer = TrajectoryBuffer()
    batch_size = config["batch_size"]
    episode_reward_list = []

    save_policy = config["save_policy"]
    
    # Training setup
    logger.debug("Training start")
    if not mp_flag:
        status_bar = tqdm(total=n_episodes, desc="Training iterations", position=0, leave=False)
        reward_log = tqdm(total=0, position=1, bar_format='{desc}', leave=False)
    # Only allocate noise arrays once
    for i in range(n_episodes):
        state = env.reset()
        done = False
        ep_reward = 0
        t = 0
        while not done:
            mean, var = actor(torch.tensor(state))
            mean, var = mean.detach().numpy(), var.detach().numpy()
            action = np.clip(np.random.multivariate_normal(mean, np.diag(var)), -1, 1)
            next_state, reward, done, _ = env.step(action)
            buffer.append((state, action, reward, next_state, done))
            ep_reward += reward
            state = next_state
            t += 1
        episode_reward_list.append(ep_reward)

        # Training
        states, actions, rewards, _, _ = buffer.sample()
        states, actions, rewards = (torch.tensor(x) for x in [states, actions, rewards])
        with torch.no_grad():
            mean, variance = actor(states)
            pi_theta_old = gaussian_pdf(actions, mean, variance)
        gamma_series = np.array([gamma**t for t in range(len(rewards)-1, -1, -1)])
        g_t = torch.tensor(np.convolve(rewards, gamma_series)[len(gamma_series)-1:], dtype=torch.float32)
        for _ in range(train_iterations):
            # Update Critic
            critic_optim.zero_grad()
            v_critic = critic(states).squeeze()
            loss = nn.functional.mse_loss(g_t, v_critic)
            loss.backward()
            critic_optim.step()
            
            # Advantage estimation
            actor_optim.zero_grad()
            advantage = g_t - critic(states).squeeze()
            mean, var = actor(states)
            pi_theta = gaussian_pdf(actions, mean, var)
            loss = ppo_clip_loss(pi_theta, pi_theta_old, advantage, epsilon)
            loss.backward()
            actor_optim.step()
            
        buffer.clear()
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
        path = Path(__file__).resolve().parent
        logger.debug(f"Saving models to {path}")
        torch.save(actor.state_dict(), path / "lunarlander_ppo_actor.pt")
        torch.save(critic.state_dict(), path / "lunarlander_ppo_critic.pt")
        logger.debug(f"Saving complete")
    
    total_reward = 0
    for _ in range(10):
        state = env.reset()
        done = False
        while not done:
            mean, _ = actor(torch.tensor(state))
            mean = mean.detach().numpy()
            next_state, reward, done, _  = env.step(mean)
            total_reward += reward
            state = next_state
        env.close()
    logger.info("Final average reward over 10 runs: {:.2f}".format(total_reward/10))

    result.put(episode_reward_list)
    return


def demo_agent():
    env = gym.make("LunarLanderContinuous-v2")
    actor = PPOActor(len(env.observation_space.low), len(env.action_space.low))
    actor.load_state_dict(torch.load(Path(__file__).parent / "lunarlander_ppo_actor.pt"))
    actor.eval()
    state = env.reset()
    done = False
    while not done:
        mean, _ = actor(torch.tensor(state))
        mean = mean.detach().numpy()
        next_state, _, done, _  = env.step(mean)
        state = next_state
        env.render()
        time.sleep(0.01)
    env.close()
    

if __name__ == "__main__":
    train = False
    demo = True
    if train:
        logging.basicConfig()
        logger.setLevel(logging.INFO)
        logger.debug("Loading config")
        path = Path(__file__).resolve().parents[1] / "config" / "experiment_config.yaml"
        with open(path, "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        logger.debug("Loading successful, starting experiment")
        result = Queue()
        lunarlander_ppo(config["lunarlander_ppo"], result)
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
    if demo:
        demo_agent()
