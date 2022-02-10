import torch
import functools
from ddpg import DDPG, DDPGActor, DDPGCritic
from noise import OrnsteinUhlenbeckNoise
import time
    
    
def timeit(_func=None, *, n=10):
    def decorator_repeat(func):
        @functools.wraps(func)
        def timeit_wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            for _ in range(n):
                func(*args, **kwargs)
            t1 = time.perf_counter()
            print(f"Finished {n} runs of {func.__name__!r}. Average execution time {(t1-t0)/n:.6f} secs")
        return timeit_wrapper
        
    if _func is None:
        return decorator_repeat
    return decorator_repeat(_func)

@timeit(n=1000)
def test_action(ddpg, state):
    return ddpg.action(state)

@timeit(n=1000)
def test_actor_learning(ddpg, batch):
    ddpg.train_actor(batch)

@timeit(n=1000)
def test_critic_learning(ddpg, batch):
    ddpg.train_critic(batch)
    ddpg.update_targets()
    
if __name__ == "__main__":
    n_states = 8
    n_actions = 2
    batch_size = 16
    actor = DDPGActor(n_states, n_actions)
    actor_t = DDPGActor(n_states, n_actions)
    critic = DDPGCritic(n_states, n_actions)
    critic_t = DDPGCritic(n_states, n_actions)
    noise_process = OrnsteinUhlenbeckNoise(0.15, 0.2, n_actions)
    ddpg = DDPG(actor, actor_t, critic, critic_t, 0.0001, 0.001, 0.99, 0.99, noise_process, action_clip = (-1.,1.), actor_clip=1., critic_clip=1.)
    ddpg.cuda()
    dev = torch.device("cuda")
    states = torch.rand(batch_size,n_states)
    state = ddpg.sanitize_array(torch.rand(1,n_states))
    actions = torch.rand(batch_size, n_actions)
    next_states = torch.rand(batch_size, n_states)
    rewards = torch.rand(batch_size, 1)
    dones = (torch.rand(batch_size, 1)>0.5).float()
    batch = [ddpg.sanitize_array(x) for x in [states, actions, rewards, next_states, dones]]
    test_action(ddpg, state)
    test_actor_learning(ddpg, batch)
    test_critic_learning(ddpg, batch)