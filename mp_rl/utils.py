import numpy as np
import torch
import torch.nn as nn
import gym


def soft_update(network: nn.Module, target: nn.Module, tau: float) -> nn.Module:
    """Performs a soft update of the target network's weights.
    
    Shifts the weights of the ``target`` by a factor of ``tau`` into the direction of the 
    ``network``.
    
    Args:
        network (nn.Module): Network from which to copy the weights.
        target (nn.Module): Network that gets updated.
        tau (float): Controls how much the weights are shifted. Valid in [0, 1].
        
    Returns:
        target (nn.Module): The updated target network.
    """
    target_state = target.state_dict()
    for k, v in network.state_dict().items():
        target_state[k] = (1 - tau)  * target_state[k]  + tau * v
    target.load_state_dict(target_state)
    return target


def test_actor(actor: nn.Module, env: gym.Env, dev: torch.device) -> float:
    """Tests an actor in its environment for 5 episodes and returns the average episode reward.
    
    Args:
        actor (nn.Module): The actor network.
        env (gym.Env): The environment for the actor.
        dev (torch.device): The device on which the ``actor`` network and `torch.Tensors` operate.
        
    Returns:
        The average reward from 5 episodes.
    """
    actor.eval()
    total_reward = 0.
    for _ in range(5):
        state = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                action = actor(torch.unsqueeze(torch.tensor(state), 0).to(dev))
                action = np.clip(action.detach().squeeze().cpu().numpy(), -1, 1)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
    actor.train()
    return total_reward/5


def running_average(values: list, window: int = 50, mode: str = 'valid') -> float:
    """Computes a running average over a list of values.
    
    Args:
        values (list): List of values that get smoothed.
        window (int, optional): Averaging window size.
        mode (str, optional): Modes for the convolution operation.
    """
    return np.convolve(values, np.ones(window)/window, mode=mode)
