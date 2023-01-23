import numpy as np
import scipy.signal
import torch
import torch.nn as nn


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


''' Replay buffer is a trick for making off policy algorithms work. '''
class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for off policy agents.
    """
    def __init__(self, obs_dim, act_dim, size):
        self.max_size = size
        self.current_size = 0        
        self.ptr = 0
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        
    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1)%self.max_size
        self.current_size = min(self.current_size+1, self.max_size)
        
    def sample_batch(self, batch_size=32):
        rand_idxs = np.random.randint(low=0, high=self.current_size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[rand_idxs],
            act=self.act_buf[rand_idxs],
            rew=self.rew_buf[rand_idxs],
            next_obs=self.next_obs_buf[rand_idxs],
            done=self.done_buf[rand_idxs]
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

    def get_current_size(self):
        return self.current_size