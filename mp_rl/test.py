import logging
import time
import torch
import yaml
import multiprocessing
from queue import Queue
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import gym
import torch.nn as nn
from tqdm import tqdm
from replay_buffer import MemoryBuffer
from utils import running_average
from ddpg import DDPG, DDPGActor, DDPGCritic, load_ddpg
from noise import OrnsteinUhlenbeckNoise


logger = logging.getLogger(__name__)


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
    
    noise_process = OrnsteinUhlenbeckNoise(mu=noise_mu, sigma=noise_sigma, dims=n_actions)
    actor = DDPGActor(n_states, n_actions)
    critic = DDPGCritic(n_states, n_actions)
    ddpg = DDPG(actor, critic, actor_lr, critic_lr, tau, gamma, noise_process, action_clip=(-1.,1.),
                actor_clip=1., critic_clip=1.)

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
        ddpg.noise_process.reset()
        ep_reward = 0
        t = 0
        t_env = 0
        t_critic = 0
        t_actor = 0
        while not done:
            t0 = time.perf_counter()
            action = ddpg.action(torch.unsqueeze(torch.as_tensor(state), 0))[0]
            next_state, reward, done, _ = env.step(action)
            buffer.append((state, action, reward, next_state, done))
            ep_reward += reward
            state = next_state
            # Training
            batch = buffer.sample(batch_size)
            t1 = time.perf_counter()
            ddpg.train_critic(batch)
            t2 = time.perf_counter()
            if t % update_delay == 0:
                ddpg.train_actor(batch)
                ddpg.update_targets()
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
        ddpg.save(path / "lunarlander_ddpg_actor.pt", path / "lunarlander_ddpg_critic.pt", path / "ddpg.pkl")
        logger.debug(f"Saving complete")
    
    total_reward = 0
    ddpg.explore = False
    for _ in range(10):
        state = env.reset()
        done = False
        while not done:
            action = ddpg.action(torch.unsqueeze(torch.as_tensor(state), 0))[0]
            next_state, reward, done, _  = env.step(action)
            total_reward += reward
            state = next_state
        env.close()
    logger.info("Final average reward over 10 runs: {:.2f}".format(total_reward/10))
    result.put(episode_reward_list)
    return


def demo_agent():
    env = gym.make("LunarLanderContinuous-v2")
    ddpg = load_ddpg(Path(__file__).parent / "ddpg.pkl")
    ddpg.explore = False
    state = env.reset()
    done = False
    while not done:
        action = ddpg.action(torch.unsqueeze(torch.as_tensor(state), 0))[0]
        next_state, _, done, _  = env.step(action)
        state = next_state
        env.render()
        time.sleep(0.02)
    env.close()
    

if __name__ == "__main__":
    train = True
    demo = True
    if train:
        logging.basicConfig()
        logger.setLevel(logging.INFO)
        logger.debug("Loading config")
        path = Path(__file__).resolve().parent / "config" / "experiment_config.yaml"
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
        plt.savefig(Path(__file__).parent / "reward.png")
        plt.show()
    if demo:
        demo_agent()
