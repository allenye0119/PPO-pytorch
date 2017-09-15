#!/usr/bin/env python

import os
import sys
from box import Box

import numpy as np
import torch
import gym

from model import Model
from trainer import Trainer


def print_config(config, d=0):
    tabs = ' ' * d * 4
    for k in config.keys():
        if isinstance(config[k], Box):
            print('{}{}:'.format(tabs, k))
            print_config(config[k], d + 1)
        else:
            print('{}{}: {}'.format(tabs, k, config[k]))

if __name__ == '__main__':
    print('Loading config...')
    config_path = sys.argv[1]
    with open(config_path, 'r') as config_f:
        config = Box.from_yaml(config_f)
    print_config(config)

    print('Setting random seed...')
    np.random.seed(config.random_seed)
    torch.cuda.manual_seed(config.random_seed)

    print('Creating environment...')
    env = gym.make(config.environment)

    print('Creating model...')
    config.model.policy.input_size = env.observation_space.shape[0]
    # config.model.policy.output_size = env.action_space.shape[0]
    config.model.policy.output_size = env.action_space.n
    config.model.value.input_size = env.observation_space.shape[0]
    model = Model(config.model)

    print('Start training...')
    trainer = Trainer(config.train, env, model, config.model)
    trainer.start()
