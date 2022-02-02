from operator import itemgetter
import argparse
import torch.multiprocessing as mp
import time
import logging
from pathlib import Path
from tqdm import tqdm
import yaml
import torch
import torch.nn as nn
import gym
import numpy as np
from shared_replay_buffer import SharedReplayBuffer
from networks import DDPGActor, DDPGCritic


logger = logging.getLogger(__name__)


def _run_env(config: dict, actor: nn.Module, queue: SharedReplayBuffer, finish_event: mp.Event, 
             train_barrier: mp.Barrier, join_event: mp.Event, pid: int):
    logger = logging.getLogger(__name__ + f":Process{pid}")
    logger.debug(f"Startup successful")
    # Setup constants, hyperparameters, bookkeeping
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("LunarLanderContinuous-v2")

    n_actions = len(env.action_space.low)
    noise_mu, noise_sigma = itemgetter("noise_mu", "noise_sigma")(config)
    
    while True:
        state = env.reset()
        done = False
        t = 0
        noise = np.zeros(n_actions)
        t_action = t_step = t_queue = 0
        while not done:
            # Sample noisy action
            t0 = time.perf_counter()
            noise = -noise*noise_mu + np.random.randn(n_actions)*noise_sigma + noise_mu
            with torch.no_grad():
                action = np.clip(actor(torch.tensor(state).to(dev)).detach().cpu().numpy() + noise, -1, 1)
            t1 = time.perf_counter()
            next_state, reward, done, _ = env.step(action)
            t2 = time.perf_counter()
            queue.put((state, action, reward, next_state, done))
            t3 = time.perf_counter()
            t_action += (t1-t0)
            t_step += (t2-t1)
            t_queue += (t3-t2)
            state = next_state
            t += 1
        finish_event.set()
        logger.debug("Finished episode, event flag set. Timings: t_action: {:.2f}, t_step: {:.2f}, t_queue: {:.2f}".format(t_action, t_step, t_queue))
        train_barrier.wait()
        if join_event.is_set():
            break


def soft_update(network: nn.Module, target: nn.Module, tau: float) -> nn.Module:
    target_state = target.state_dict()
    for k, v in network.state_dict().items():
        target_state[k] = (1 - tau)  * target_state[k]  + tau * v
    target.load_state_dict(target_state)
    return target


def test_actor(actor, env, dev):
    total_reward = 0.
    for _ in range(5):
        state = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                action = np.clip(actor(torch.tensor(state.to(dev))).detach().cpu().numpy(), -1, 1)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
    return total_reward/5


def main(args):
    # Setup logging and load configs
    logging.basicConfig()
    logging.getLogger().setLevel(args.loglvl)
    logger.debug("Loading config")
    path = Path(__file__).resolve().parent / "config" / "experiment_config.yaml"
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)["lunarlander_ddpg"]
    logger.debug("Loading successful, starting experiment")
    n_workers = args.nprocesses
    
    env = gym.make("LunarLanderContinuous-v2")  # Gym is only created for its constants
    n_states = len(env.observation_space.low)
    n_actions = len(env.action_space.low)
    # Unpack config
    actor_lr, critic_lr, gamma, tau = itemgetter("actor_lr", "critic_lr", "gamma", "tau")(config)
    update_delay, n_episodes, n_train = itemgetter("update_delay", "n_episodes", "n_train")(config)
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
    
    logger.debug("Initializing and starting shared replay buffer")
    
    # Create shared replay buffer, worker threads and events
    # All worker processes and the training process have to wait for the training barrier 
    buffer = SharedReplayBuffer(config["buffer_size"], n_workers)
    train_barrier = mp.Barrier(n_workers+1)
    join_event = mp.Event()
    logger.debug("Creating env workers")
    workers = []
    for pid in range(n_workers):
        p = mp.Process(target=_run_env, args=(config, actor, buffer.queue, buffer._prod_events[pid],
                                              train_barrier, join_event, pid))
        workers.append(p)
        p.start()
    # Start buffer after processes. Recommended to not get processes mixed up
    buffer.start()  # Start consumer thread for the queue
    
    status_bar = tqdm(total=n_episodes, desc="Training iterations", position=0)
    reward_log = tqdm(total=0, position=1, bar_format='{desc}')
    for ep in range(n_episodes):
        buffer.start()
        t0 = time.perf_counter()
        buffer.join()
        t01 = time.perf_counter()
        logger.debug("Buffer join successful")
        t_agent = 0
        t_critic = 0
        logger.debug("Starting training")
        for t in range(n_train):
            # Training
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
            t_critic += t2-t1
            t_agent += t3-t2
        train_barrier.wait()
        status_bar.update()
        logger.debug("Sample time: {:2f}, Critic train time: {:2f}, Actor train time: {}".format(t01-t0,t_agent, t_critic))
        if ep % 10 == 0:
            logger.debug("Testing actor network")
            reward = test_actor(actor, env, dev)
            reward_log.set_description_str("Current running average reward: {:.1f}".format(reward))
        if reward > 200:
            logger.info("Training succeeded.")
            break
    # Join event is set before waiting for train_barrier to ensure worker processes receive shutdown
    # upon passing the barrier
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
        

if __name__ == "__main__":
    mp.set_start_method('spawn')  # Cuda cannot re-initialize in forked subprocesses
    parser = argparse.ArgumentParser()
    parser.add_argument('--nprocesses', help='Number of worker threads for sample generation',
                        default=5)
    parser.add_argument('--loglvl', help="Logger levels", choices=["DEBUG", "INFO", "WARN", "ERROR"],
                        default="INFO")
    args = parser.parse_args()
    if args.loglvl == "DEBUG":
        args.loglvl = logging.DEBUG
    elif args.loglvl == "INFO":
        args.loglvl = logging.INFO
    elif args.loglvl == "WARN":
        args.loglvl = logging.WARN
    elif args.loglvl == "ERROR":
        args.loglvl = logging.ERROR
    main(args)
