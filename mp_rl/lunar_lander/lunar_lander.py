from operator import itemgetter
import argparse
import torch.multiprocessing as mp
import time
import logging
from pathlib import Path
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import gym
import numpy as np
from shared_replay_buffer import SharedReplayBuffer
from utils import soft_update, test_actor, running_average
from .networks import DDPGActor, DDPGCritic


logger = logging.getLogger(__name__)


def _run_env(config: dict, actor: nn.Module, queue: mp.Queue, finish_event: mp.Event, 
             train_barrier: mp.Barrier, join_event: mp.Event, pid: int):
    """Worker process loop that generates experience samples from an agent.
    
    The loop is interrupted by the ``join_event`` from the main process to keep processes alive as
    long as possible. This avoids costly reinitialization of processes after each training cycle.
    
    Args:
        config (dict): Dictionary with training parameters.
        actor (nn.Module): The actor network that is used to select actions. Note that the network
            weights have to be in shared memory for all processes to include changes from the main 
            training process!
        queue (mp.Queue): Multiprocessing queue to insert experience samples into.
        finish_event (mp.Event): Event signal for finishing a cycle.
        train_barrier (mp.Barrier): A barrier that syncronizes the main process with all worker
            processes and makes sure the workers exit after the main process decides to stop them.
        join_event (mp.Event): Event signal from the main process to terminate the experience 
            generation loop.
    """
    logger = logging.getLogger(__name__ + f":Process{pid}")
    logger.debug(f"Startup successful")
    # Setup constants, hyperparameters, bookkeeping
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor.eval()
    env = gym.make("LunarLanderContinuous-v2")
    n_actions = len(env.action_space.low)
    noise_mu, noise_sigma = itemgetter("noise_mu", "noise_sigma")(config)

    while True:  # Experience collection loop
        for _ in range(config["episodes"]):  # Each process can have multiple rollouts
            state = env.reset()
            done = False
            t = 0
            noise = np.zeros(n_actions)
            while not done:
                # Sample noisy action
                noise = -noise*noise_mu + np.random.randn(n_actions)*noise_sigma + noise_mu
                with torch.no_grad():
                    action = actor(torch.unsqueeze(torch.tensor(state), 0).to(dev))
                    action = np.clip(action.detach().squeeze().cpu().numpy() + noise, -1, 1)
                next_state, reward, done, _ = env.step(action)
                queue.put((state, action, reward, next_state, done))
                state = next_state
                t += 1
        finish_event.set()
        logger.debug("Finished episodes, event flag set.")
        train_barrier.wait()
        if join_event.is_set():
            break


def lunar_lander_ddpg(args: argparse.Namespace):
    """Solves the LunarLander-Continuous-v2 environment from the OpenAI gym.
    
    Creates worker processes, shares the actor network weights and trains actor/critic networks with
    their target networks for multiple episodes. Optionally saves the networks' weights.
    """
    # Setup logging and load configs
    logger.debug("Loading config")
    path = Path(__file__).resolve().parents[1] / "config" / "experiment_config.yaml"
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)["lunarlander_ddpg"]
    logger.debug("Loading successful, starting experiment")
    n_workers = args.nprocesses
    env = gym.make("LunarLanderContinuous-v2")  # Gym is only created for its constants
    n_states = len(env.observation_space.low)
    n_actions = len(env.action_space.low)
    episode_reward_list = []
    # Unpack config
    actor_lr, critic_lr, gamma, tau = itemgetter("actor_lr", "critic_lr", "gamma", "tau")(config)
    update_delay, epochs, cycles = itemgetter("update_delay", "epochs", "cycles")(config)
    train_episodes = itemgetter("train_episodes")(config)
    logger.debug("Config unpack successful")
    
    # Initialize actor critic networks and optimizers
    logger.debug("Initializing actor critic networks")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(f"Device in use: {dev}")
    actor = DDPGActor(n_states, n_actions).to(dev)
    actor.share_memory()
    critic = DDPGCritic(n_states, n_actions).to(dev)
    target_actor = DDPGActor(n_states, n_actions).to(dev)
    target_critic = DDPGCritic(n_states, n_actions).to(dev)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    logger.debug("Actor critic network initialization successful")
    
    # Create shared replay buffer, worker processes, events and barriers
    logger.debug("Initializing and starting shared replay buffer")
    buffer = SharedReplayBuffer(config["buffer_size"], n_workers)
    train_barrier = mp.Barrier(n_workers+1)  # All workers and the training process
    join_event = mp.Event()
    logger.debug("Creating env workers")
    workers = []
    for pid in range(n_workers):
        p = mp.Process(target=_run_env, args=(config, actor, buffer.queue, buffer._prod_events[pid],
                                              train_barrier, join_event, pid))
        workers.append(p)
        p.start()
    
    # Status bars show remaining execution time, current reward etc.
    status_bar = tqdm(total=epochs*cycles, desc="Training iterations", position=0)
    reward_log = tqdm(total=0, position=1, bar_format='{desc}')
    try:
        for ep in range(epochs*cycles):
            buffer.start()  # Consumer starts working on the queue
            t0 = time.perf_counter()
            buffer.join()  # Waits for the workers, then finishes consuming the queue and returns
            t01 = time.perf_counter()
            logger.debug("Buffer join successful")
            t_agent = 0
            t_critic = 0
            logger.debug("Starting training")
            actor.train()
            # Training loop for actor, critic and target networks
            for t in range(train_episodes):
                # Update the critic with batches sampled from the buffer
                t1 = time.perf_counter()
                critic_optim.zero_grad()
                states, actions, rewards, next_states, dones = buffer.sample(config["batch_size"])
                next_states = torch.tensor(next_states).to(dev)
                with torch.no_grad():
                    next_actions = target_actor(next_states)
                    next_q = target_critic(next_states, next_actions).squeeze()
                    rewards = torch.tensor(rewards, requires_grad=False).to(dev)
                    dones = 1 - torch.tensor(dones, requires_grad=False).to(dev)
                    rewards = rewards + gamma * next_q * dones
                
                states = torch.tensor(states).to(dev)
                actions = torch.tensor(actions).to(dev)
                q_actions = critic(states, actions).squeeze()
                
                loss = nn.functional.mse_loss(q_actions, rewards)
                loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.)
                critic_optim.step()
                t2 = time.perf_counter()
                # Update the actor. Less frequent updates to the policy stabilize training
                # TODO: Not necessarily the case for mp_rl, check OpenAI paper for details
                if t % update_delay == 0:
                    actor_optim.zero_grad()
                    actions = actor(states)
                    next_q = critic(states, actions)
                    loss = -torch.mean(next_q)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.)
                    actor_optim.step()
                t3 = time.perf_counter()
                t_critic += t2-t1
                t_agent += t3-t2
            train_barrier.wait()
            status_bar.update()
            # Target updates are only performed once every cycle with soft updates
            target_actor = soft_update(actor, target_actor, tau)
            target_critic = soft_update(critic, target_critic, tau)
            logger.debug("Sample time: {:2f}, Critic train time: {:2f}, Actor train time: {}".format(t01-t0,t_agent, t_critic))
            # Test the current agent policy every 10 episodes
            if ep % 10 == 0:
                logger.debug("Testing actor network")
                reward = test_actor(target_actor, env, dev)
                episode_reward_list.append(reward)
                reward_log.set_description_str("Current average reward: {:.1f}".format(reward))
            # Early stopping if the agent achieved a good policy
            if reward > 200:
                logger.info("Training succeeded.")
                break

        # Graceful process shutdown. Buffer join is invoked to make sure all processes have passed
        # the join_event.is_set() call from the previous run and are waiting on the barrier again. 
        # Otherwise the barrier can deadlock on processes that exit after an early read of the 
        # event. This means one cycle is performed and discarded, but this is negligible. Join event
        # is then set before waiting for train_barrier to ensure worker processes receive the 
        # shutdown upon passing the barrier. 
        buffer.start()
        buffer.join()
        join_event.set()
        train_barrier.wait()
        for p in workers:
            p.join()
        logger.info("Training finished")
        
        if config["save_policy"]:
            path = Path(__file__).resolve().parent
            logger.debug(f"Saving models to {path}")
            torch.save(actor.state_dict(), path / "lunarlander_ddpg_actor.pt")
            torch.save(critic.state_dict(), path / "lunarlander_ddpg_critic.pt")
            logger.debug(f"Saving complete")

    except KeyboardInterrupt:  # Enable to save the plot of unfinished trainings
        pass
    
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
