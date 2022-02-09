import logging
import time
from pathlib import Path
import argparse
import yaml
import matplotlib.pyplot as plt
from multiprocessing import Queue
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from memory import MemoryBuffer
import gym
import inspect
from gym.wrappers import FilterObservation, FlattenObservation


logger = logging.getLogger(__name__)


def running_average(values: list, window: int = 50, mode: str = 'valid'):
    return np.convolve(values, np.ones(window)/window, mode=mode)


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
            action = 2*(np.random.rand(4)-0.5)
            next_state, reward, done, _ = env.step(action)
            buffer.append((state, action, reward, next_state, done))
            state = next_state
    logger.debug("Buffer filled")


class DDPGActor(nn.Module):
    
    def __init__(self, n_obs: int, n_actions: int):
        super().__init__()
        self.l1 = nn.Linear(n_obs, 400)
        self.l2 = nn.Linear(400, 200)
        self.l3 = nn.Linear(200, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return torch.tanh(self.l3(x))
    
class DDPGCritic(nn.Module):

    def __init__(self, n_obs: int, n_actions: int):
        super().__init__()
        self.l1 = nn.Linear(n_obs, 400)
        self.l2 = nn.Linear(400+n_actions, 200)
        self.l3 = nn.Linear(200, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state = torch.relu(self.l1(state))
        x = torch.relu(self.l2(torch.cat([state, action], dim=1)))
        return self.l3(x)


def train(config: dict, result: Queue):
    logger.debug(f"Config: {config}")
    # Setup constants, hyperparameters, bookkeeping
    env = gym.make("FetchReach-v1", reward_type="dense")
    env = FlattenObservation(FilterObservation(env, filter_keys=["observation", "desired_goal"]))
    n_episodes = config["n_episodes"]
    n_obs = len(env.observation_space.low)
    n_actions = len(env.action_space.low)
    logger.debug(env.action_space)
    gamma = config["gamma"]
    actor_lr = config["actor_lr"]
    critic_lr = config["critic_lr"]
    noise_mu = config["noise_mu"]
    noise_sigma = config["noise_sigma"]
    tau = config["tau"]
    update_delay = config["update_delay"]
    
    actor = DDPGActor(n_obs, n_actions)
    critic = DDPGCritic(n_obs, n_actions)
    target_actor = DDPGActor(n_obs, n_actions)
    target_critic = DDPGCritic(n_obs, n_actions)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    buffer = MemoryBuffer(config["buffer_size"])
    fill_buffer(env, buffer, config["buffer_size"]/10)
    batch_size = config["batch_size"]
    episode_reward_list = []

    save_policy = config["save_policy"]
    
    # Training setup
    logger.debug("Training start")
    status_bar = tqdm(total=n_episodes, desc="Training iterations", position=0, leave=False)
    reward_log = tqdm(total=0, position=1, bar_format='{desc}', leave=False)
    try:
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
                noise = -noise*noise_mu + np.random.randn(n_actions)*noise_sigma
                with torch.no_grad():
                    action = np.clip(actor(torch.tensor(state)).detach().numpy() + noise, -1, 1)
                next_state, reward, done, _ = env.step(action)
                if t < 49 and done:
                    logger.info("SOLVED TASK")
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
            reward_log.set_description_str("Current running average reward: {:.1f}".format(av_reward))
            status_bar.update()
            if i > 50 and av_reward > -0.5:
                logger.debug("Training stopped, agent has solved the task.")
                break
        env.close()
        logger.debug("Training has ended")
    except KeyboardInterrupt:
        pass

    if save_policy:
        path = Path(__file__).resolve().parent
        logger.debug(f"Saving models to {path}")
        torch.save(actor.state_dict(), path / "fetchreach_ddpg_actor.pt")
        torch.save(critic.state_dict(), path / "fetchreach_ddpg_critic.pt")
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
    env = gym.make("FetchReach-v1")
    env = FlattenObservation(FilterObservation(env, filter_keys=["observation", "desired_goal"]))
    actor = DDPGActor(len(env.observation_space.low), len(env.action_space.low))
    actor.load_state_dict(torch.load(Path(__file__).parent / "fetchreach_ddpg_actor.pt"))
    actor.eval()
    state = env.reset()
    done = False
    t = 0
    while not done:
        action = actor(torch.tensor(state)).detach().numpy()
        next_state, _, done, _  = env.step(action)
        state = next_state
        env.render()
        time.sleep(0.05)
    env.close()
    

def main(args):
    train = False
    demo = True
    logging.basicConfig()
    logger.setLevel(args.loglvl)
    if train:
        logger.debug("Loading config")
        path = Path(__file__).resolve().parents[1] / "config" / "experiment_config.yaml"
        with open(path, "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        logger.debug("Loading successful, starting experiment")
        result = Queue()
        fetch_reach_ddpg(config["fetchreach_ddpg"], result)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
