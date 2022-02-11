import logging
from pathlib import Path
from operator import itemgetter

import numpy as np
import gym
import torch
from tqdm import tqdm

from mp_rl.ddpg import DDPG, DDPGActor, DDPGCritic
from mp_rl.replay_buffer import HERBuffer
from mp_rl.noise import OrnsteinUhlenbeckNoise
from mp_rl.utils import fill_buffer, running_average, ddp_poll_shutdown, save_plots, save_stats
from mp_rl.utils import unwrap_obs


logger = logging.getLogger(__name__)


def train(rank: int, size: int, config):
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
    T = 50  # Episode length
    n_states = len(env.observation_space["observation"].low)
    n_goals = len(env.observation_space["desired_goal"].low)
    n_actions = len(env.action_space.low)
    gamma, actor_lr, critic_lr, tau = itemgetter("gamma", "actor_lr", "critic_lr", "tau")(config)

    noise_process = OrnsteinUhlenbeckNoise(mu=config["mu"], sigma=config["sigma"], dims=n_actions)
    ddpg = DDPG(DDPGActor(n_states+n_goals, n_actions), DDPGActor(n_states+n_goals, n_actions),
                DDPGCritic(n_states+n_goals, n_actions), DDPGCritic(n_states+n_goals, n_actions),
                actor_lr, critic_lr, tau, gamma, noise_process, action_clip=(-1, 1), actor_clip=1.,
                critic_clip=1.)
    ddpg.init_ddp()
    logger.info(f"P{rank}: DDPG moved to DDP, filling buffer")
    buffer = HERBuffer(n_states, n_actions, n_goals, T, 4, config["buffer_size"], "default")
    fill_buffer(buffer, env)
    logger.info(f"P{rank}: Buffer filled, starting training")
    if rank == 0:
        status_bar = tqdm(total=config["epochs"]*config["cycles"], desc="Training iterations",
                          position=0, leave=False)
        reward_log = tqdm(total=0, position=1, bar_format='{desc}', leave=False)
        ep_rewards = []
        ep_lengths = []
    try:
        def _train():
            for epoch in range(config["epochs"]*config["cycles"]):
                for episode in range(config["episodes"]):
                    obs = env.reset()
                    state, goal, agoal = unwrap_obs(obs)
                    done = False
                    ddpg.noise_process.reset()
                    ep_reward = 0
                    t = 0
                    ep_buffer = buffer.get_trajectory_buffer()
                    while not done:
                        f_state = np.concatenate((state, goal))
                        action = ddpg.action(torch.unsqueeze(torch.as_tensor(f_state), 0))[0]
                        next_obs, reward, done, _ = env.step(action)
                        next_state, next_goal, next_agoal = unwrap_obs(next_obs)
                        for key, val in zip(["s", "a", "r", "sn", "d", "g", "ag"],
                                            [state, action, reward, next_state, done, goal, agoal]):
                            ep_buffer[key][t] = val
                        ep_reward += reward
                        t += 1
                        state, goal, agoal = next_state, next_goal, next_agoal
                    buffer.append(ep_buffer)
                    if rank == 0:
                        ep_rewards.append(ep_reward)
                        ep_lengths.append(t)
                # Training
                for train_episode in range(config["train_episodes"]):
                    dbatch = buffer.sample(config["batch_size"])
                    f_state, f_next_state = [np.concatenate((dbatch[key], dbatch["g"]), axis=1)
                                             for key in ("s", "sn")]
                    batch = [f_state, dbatch["a"], dbatch["r"], f_next_state, dbatch["d"]]
                    batch = [ddpg.sanitize_array(x) for x in batch]
                    ddpg.train_critic(batch)
                    ddpg.train_actor(batch)
                ddpg.update_targets()
                if rank == 0:
                    av_reward = running_average(ep_rewards)[-1]
                    reward_log.set_description_str("Current running average reward: {:.1f}".format(av_reward))  # noqa: E501
                    status_bar.update()
                    # Rank 0 process signals early stopping. ddp_poll_shutdown is called in both
                    # cases because all_reduce is synchronized across processes (blocks otherwise)
                    # >50 check because initial average is skewed towards 0, %10 to reduce amount of
                    # expensive all_reduce calls
                    if epoch > 50 and epoch % 10 == 0:
                        if av_reward > -1.:
                            logger.info("Problem solved, returning from all tasks")
                            ddp_poll_shutdown(True)
                            return
                        ddp_poll_shutdown()
                else:  # Graceful shutdown for processes of rank > 0
                    if epoch > 50 and epoch % 10 == 0 and ddp_poll_shutdown():
                        return
            return

        _train()
    except KeyboardInterrupt:  # Enable saves on interrupts
        pass

    if rank == 0 and config["save_policy"]:
        path = Path(__file__).parent
        logger.debug(f"Saving models to {path}")
        ddpg.save(path / "fetchreach_actor.pt", path / "fetchreach_critic.pt", path / "ddpg.pkl")
        logger.debug("Saving complete")

    if rank == 0:
        save_plots(ep_rewards, ep_lengths, Path(__file__).parent / "stats.png", window=50)
        save_stats(ep_rewards, ep_lengths, Path(__file__).parent / "stats.png", window=50)
