import logging
from pathlib import Path
import string
import random
import yaml
import matplotlib.pyplot as plt
from chons.experiments.utils import running_average
from chons.core.mp_engine import MPEngine
from chons.experiments.cartpole import cartpole


if __name__ == "__main__":
    # Logging setup
    logging.basicConfig()
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        if "chons" in logger.name:
            logger.setLevel(logging.INFO)
    
    # Config
    nruns = 4
    nprocesses = 2
    letters = string.ascii_lowercase
    uuid = ( ''.join(random.choice(letters) for _ in range(5)))
    root = Path(__file__).parent
    (root / "results" / uuid).mkdir(parents=True, exist_ok=False)  # Crash on UUID collision
    config = {'cartpole': {'n_episodes': 5000,
                           'gamma': 1,
                           'lr': 0.001,
                           'eps_max': 1.,
                           'eps_min': 0.01,
                           'buffer_size': 5000,
                           'batch_size': 256,
                           'save_policy': False}}

    with open(root / "results" / uuid / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    engine = MPEngine(nprocesses)
    engine.register_experiment(cartpole, root / "results" / uuid / "config.yaml")
    results = []
    for i in range(nruns):
        rqueue = engine.run()
        while not rqueue.empty():
            results.append(rqueue.get())

    for (idx, episode_reward_list) in enumerate(results):
        fig, ax = plt.subplots()
        ax.plot(episode_reward_list)
        smooth_reward = running_average(episode_reward_list)
        index = range(len(episode_reward_list)-len(smooth_reward), len(episode_reward_list))
        ax.plot(index, smooth_reward)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Accumulated reward')
        ax.set_title('Agent performance over time')
        ax.legend(["Episode reward", "Running average reward"])
        plt.savefig(root / "results" / uuid / f'run_{idx}.png')
