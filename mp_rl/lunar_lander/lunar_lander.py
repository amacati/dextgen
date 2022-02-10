import logging
from pathlib import Path
from operator import itemgetter

import gym
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from mp_rl.ddpg import DDPG, DDPGActor, DDPGCritic
from mp_rl.replay_buffer import MemoryBuffer
from mp_rl.noise import OrnsteinUhlenbeckNoise
from mp_rl.utils import fill_buffer, running_average, ddp_poll_shutdown


logger = logging.getLogger(__name__)


def train(rank:int, size: int, config):
    """Training function for the lunar lander continuous gym.
    
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
    config = config["lunarlander_ddpg"]
    logger.debug(f"Config: {config}")
    # Setup constants, hyperparameters, bookkeeping
    env = gym.make("LunarLanderContinuous-v2")
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
        def _train():  # Function to break out of all loops in early stopping polls
            for epoch in range(config["epochs"]*config["cycles"]):
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
                    # Rank 0 process signals early stopping. ddp_poll_shutdown is called in both 
                    # cases because all_reduce is synchronized across processes (blocks otherwise)
                    # >50 check because initial average is skewed towards 0, %10 to reduce amount of
                    # expensive all_reduce calls
                    if epoch > 50 and epoch % 10 == 0:
                        if av_reward > -150:
                            logger.info("Problem solved, returning from all tasks")
                            ddp_poll_shutdown(True)
                            return
                        ddp_poll_shutdown()
                else:  # Graceful shutdown for processes of rank > 0
                    if epoch > 50 and epoch % 10 == 0 and ddp_poll_shutdown():
                        return
            return

        _train()                    
    except KeyboardInterrupt:  # Enable parameter save and plot save after abort
        pass

    if rank == 0 and config["save_policy"]:
        path = Path(__file__).parent
        logger.debug(f"Saving models to {path}")
        ddpg.save(path / "lunarlander_ddpg_actor.pt", path / "lunarlander_ddpg_critic.pt", path / "ddpg.pkl")
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
