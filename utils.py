import numpy as np
import torch
from torch.utils.data import Dataset
from IPython import embed

class MemoryDataset(Dataset):
    def __init__(self):
        self.observations = [] # cuda tensor of shape (1,) + env.observation_space.shape
        self.actions = []      # cuda tensor of shape (1,) + shape env.action_space.shape
        self.rewards = []  # cuda tensor of shape (1, )
        self.advantages = []   # cuda tensor of shape (1, )

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return {
            'observation': self.observations[idx],
            'action': self.actions[idx],
            'reward': self.rewards[idx],
            'advantage': self.advantages[idx]
        }

    def append(self, observations, actions, rewards, advantages):
        self.observations += observations
        self.actions += actions
        self.rewards += rewards
        self.advantages += advantages

    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.advantages = []

def collate_fn(batch):
    observation = torch.cat([b['observation'] for b in batch])
    # action = torch.stack([b['action'] for b in batch])
    action = [b['action'] for b in batch]
    reward = torch.cat([b['reward'] for b in batch])
    advantage = torch.cat([b['advantage'] for b in batch])

    return {
        'observation': observation,
        'action': action,
        'reward': reward,
        'advantage': advantage
    }

def normal(std, mean, x):
    a = 1 / (2 * np.pi * std ** 2).sqrt()
    b = (-((x - mean) ** 2 / (2 * std ** 2))).exp()
    prob = a * b

    return prob

def normalize(x):
    mean = x.mean()
    std = x.std(unbiased=False)
    normalized_x = (x - mean) / (std + 1e-16)

    return normalized_x

class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        # print ("push shape: ", x.shape)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        # print ("Zfilter shape: ", shape)
        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-16)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape
