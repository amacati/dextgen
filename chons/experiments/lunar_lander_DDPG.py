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
from memory import MemoryBuffer
from utils import running_average


logger = logging.getLogger(__name__)


def soft_update(network: nn.Module, target: nn.Module, tau: float) -> nn.Module:
    target_state = target.state_dict()
    for k, v in network.state_dict().items():
        target_state[k] = (1 - tau)  * target_state[k]  + tau * v
    target.load_state_dict(target_state)
    return target


def fill_buffer(env, buffer: MemoryBuffer, L: int):
    while len(buffer) < L:
        done = False
        state = env.reset()
        while not done:
            action = 2*(np.random.rand(2)-0.5)
            next_state, reward, done, _ = env.step(action)
            buffer.append((state, action, reward, next_state, done))
            state = next_state
    logger.debug("Buffer filled")


class DDPGActor(nn.Module):
    
    def __init__(self, n_states: int, n_actions: int):
        super().__init__()
        self.l1 = nn.Linear(n_states, 400)
        self.l2 = nn.Linear(400, 200)
        self.l3 = nn.Linear(200, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return torch.tanh(self.l3(x))
    
class DDPGCritic(nn.Module):
    
    def __init__(self, n_states: int, n_actions: int):
        super().__init__()
        self.l1 = nn.Linear(n_states, 400)
        self.l2 = nn.Linear(400+n_actions, 200)
        self.l3 = nn.Linear(200, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state = torch.relu(self.l1(state))
        x = torch.relu(self.l2(torch.cat([state, action], dim=1)))
        return self.l3(x)
    

def lunarlander_ddpg(config: dict, result: Queue):
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
    noise_mu = config["noise_mu"]
    noise_sigma = config["noise_sigma"]
    tau = config["tau"]
    update_delay = config["update_delay"]
    
    actor = DDPGActor(n_states, n_actions)
    critic = DDPGCritic(n_states, n_actions)
    target_actor = DDPGActor(n_states, n_actions)
    target_critic = DDPGCritic(n_states, n_actions)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    buffer = MemoryBuffer(config["buffer_size"])
    fill_buffer(env, buffer, config["buffer_size"])
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
        noise = np.zeros(n_actions)
        t_env = 0
        t_critic = 0
        t_actor = 0
        while not done:
            t0 = time.perf_counter()
            # Sample noisy action
            noise = -noise*noise_mu + np.random.randn(2)*noise_sigma
            with torch.no_grad():
                action = np.clip(actor(torch.tensor(state)).detach().numpy() + noise, -1, 1)
            next_state, reward, done, _ = env.step(action)
            buffer.append((state, action, reward, next_state, done))
            ep_reward += reward
            state = next_state
            t1 = time.perf_counter()
            # Training
            critic_optim.zero_grad()
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            next_states = torch.tensor(next_states)
            with torch.no_grad():
                next_actions = target_actor(next_states)
                next_q = target_critic(next_states, next_actions).squeeze()
                rewards = torch.tensor(rewards, requires_grad=False)
                dones = 1 - torch.tensor(dones, requires_grad=False)
                rewards = rewards + gamma * next_q * dones
            
            states = torch.tensor(states)
            actions = torch.tensor(actions)
            q_actions = critic(states, actions).squeeze()
            
            loss = nn.functional.mse_loss(q_actions, rewards)
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.)
            critic_optim.step()
            t2 = time.perf_counter()
            if t % update_delay == 0:
                actor_optim.zero_grad()
                actions = actor(states)
                next_q = critic(states, actions)
                loss = -torch.mean(next_q)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.)
                actor_optim.step()
                target_actor = soft_update(actor, target_actor, tau)
                target_critic = soft_update(critic, target_critic, tau)
            t3 = time.perf_counter()
            t += 1
            t_env += (t1-t0)
            t_critic += (t2-t1)
            t_actor += (t3-t2)
        logger.debug("t_env: {:.3f}, t_critic: {:.3f}, t_actor: {:.3f}".format(t_env, t_critic, t_actor))
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
        path = Path(__file__).resolve().parent
        logger.debug(f"Saving models to {path}")
        torch.save(actor.state_dict(), path / "lunarlander_ddpg_actor.pt")
        torch.save(critic.state_dict(), path / "lunarlander_ddpg_critic.pt")
        logger.debug(f"Saving complete")
    
    total_reward = 0
    for _ in range(10):
        state = env.reset()
        done = False
        while not done:
            action = actor(torch.tensor(state)).detach().numpy()
            next_state, reward, done, _  = env.step(action)
            total_reward += reward
            state = next_state
        env.close()
    logger.info("Final average reward over 10 runs: {:.2f}".format(total_reward/10))

    result.put(episode_reward_list)
    return


def demo_agent():
    env = gym.make("LunarLanderContinuous-v2")
    actor = DDPGActor(len(env.observation_space.low), len(env.action_space.low))
    actor.load_state_dict(torch.load(Path(__file__).parent / "lunarlander_ddpg_actor.pt"))
    actor.eval()
    state = env.reset()
    done = False
    while not done:
        action = actor(torch.tensor(state)).detach().numpy()
        next_state, _, done, _  = env.step(action)
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
        lunarlander_ddpg(config["lunarlander_ddpg"], result)
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
