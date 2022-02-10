import logging
from pathlib import Path
from operator import itemgetter

import gym
from gym.wrappers import FilterObservation, FlattenObservation
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from ddpg import DDPG, DDPGActor, DDPGCritic
from replay_buffer import MemoryBuffer
from noise import OrnsteinUhlenbeckNoise
from utils import fill_buffer, running_average


logger = logging.getLogger(__name__)


def train(rank:int, size: int, config):
    """Training function for the fetch-reach dense reward gym.
    
    Uses DDP to distribute training among several processes. Process 0 is responsible for reporting
    running stats and saving the results. Problem is solved with DDPG.
    
    Args:
        rank (int): Process rank in the DDP process group.
        size (int): Total DDP world size.
        config (dict): Config dictionary with hyperparameters.
    """

    logging.basicConfig()
    logger.setLevel(logging.INFO)
    logger.debug(f"Process {rank} startup successful.")
    config = config["fetchreach_ddpg"]
    logger.debug(f"Config: {config}")
    # Setup constants, hyperparameters, bookkeeping
    env = gym.make("FetchReach-v1", reward_type="dense")
    env = FlattenObservation(FilterObservation(env, filter_keys=["observation", "desired_goal"]))
    n_states = len(env.observation_space.low)
    n_actions = len(env.action_space.low)
    gamma, actor_lr, critic_lr, tau = itemgetter("gamma", "actor_lr", "critic_lr", "tau")(config)
    
    noise_process = OrnsteinUhlenbeckNoise(mu=config["mu"], sigma=config["sigma"], dims=n_actions)
    ddpg = DDPG(DDPGActor(n_states, n_actions), DDPGActor(n_states, n_actions), 
                DDPGCritic(n_states, n_actions), DDPGCritic(n_states, n_actions), 
                actor_lr, critic_lr, tau, gamma, noise_process, action_clip=(-1.,1.), actor_clip=1.,
                critic_clip=1.)
    ddpg.init_ddp()
    logger.info(f"P{rank}: DDPG moved to DDP, filling buffer")
    buffer = MemoryBuffer(config["buffer_size"])
    fill_buffer(env, buffer)
    logger.info(f"P{rank}: Buffer filled, starting training")
    if rank == 0:
        status_bar = tqdm(total=config["epochs"]*config["cycles"], desc="Training iterations", position=0, leave=False)
        reward_log = tqdm(total=0, position=1, bar_format='{desc}', leave=False)
        episode_reward_list = []
    try:
        for epoch in range(config["epochs"]):
            for cycle in range(config["cycles"]):
                for episode in range(config["episodes"]):
                    state = env.reset()
                    done = False
                    ddpg.noise_process.reset()
                    ep_reward = 0
                    while not done:
                        action = ddpg.action(torch.unsqueeze(torch.as_tensor(state), 0))[0]
                        next_state, reward, done, _ = env.step(action)
                        buffer.append((state, action, reward, next_state, done))
                        ep_reward += reward
                        state = next_state
                    if rank == 0:
                        episode_reward_list.append(ep_reward)
                # Training
                for train_episode in range(config["train_episodes"]):
                    batch = buffer.sample(config["batch_size"])
                    batch = [ddpg.sanitize_array(x) for x in batch]
                    ddpg.train_critic(batch)
                    ddpg.train_actor(batch)
                ddpg.update_targets()
                if rank == 0:
                    av_reward = running_average(episode_reward_list)[-1]
                    reward_log.set_description_str("Current running average reward: {:.1f}".format(av_reward))
                    status_bar.update()
    except KeyboardInterrupt:  # Enable saves on interrupts
        pass

    if rank == 0 and config["save_policy"]:
        path = Path(__file__).parent
        logger.debug(f"Saving models to {path}")
        ddpg.save(path / "fetchreach_actor.pt", path / "fetchreach_critic.pt", path / "ddpg.pkl")
        logger.debug(f"Saving complete")

    if rank == 0:    
        fig, ax = plt.subplots()
        ax.plot(episode_reward_list)
        smooth_reward = running_average(episode_reward_list, window=10)
        index = range(len(episode_reward_list)-len(smooth_reward), len(episode_reward_list))
        ax.plot(index, smooth_reward)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Accumulated reward')
        ax.set_title('Agent performance over time')
        ax.legend(["Episode reward", "Running average reward"])
        plt.savefig(Path(__file__).parent / "reward.png")
