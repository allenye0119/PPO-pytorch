import os
import shutil
from IPython import embed

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate,
            intermediate_activation, final_activation):

        super(MLP, self).__init__()

        hidden_size = [input_size] + hidden_size + [output_size]
        intermediate_activation = getattr(nn, intermediate_activation)
        final_activation = getattr(nn, final_activation, None)

        layers = []
        for i in range(len(hidden_size) - 1):
            layers.append(nn.Linear(*hidden_size[i:i + 2]))
            if dropout_rate != 0:
                layers.append(nn.Dropout(p=dropout_rate))
            if i != len(hidden_size) - 2:
                layers.append(intermediate_activation())
            elif final_activation:
                layers.append(final_activation())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# class Policy(nn.Module):
    # def __init__(self, config):
        # super(Policy, self).__init__()

        # hs1 = config.input_size * 10
        # hs3 = config.output_size * 10
        # hs2 = int(np.round(np.sqrt(hs1 * hs3)))
        # config.hidden_size = [hs1, hs2, hs3]
        # self.mean_mlp = MLP(**config)
        # self.log_std = nn.Parameter(torch.zeros(config.output_size))

    # def forward(self, x):
        # mean = self.mean_mlp(x)
        # std = self.log_std.exp()

        # return mean, std


class Policy(nn.Module):
    def __init__(self, config):
        super(Policy, self).__init__()

        hs1 = config.input_size * 10
        hs3 = config.output_size * 10
        hs2 = int(np.round(np.sqrt(hs1 * hs3)))
        config.hidden_size = [hs1, hs2, hs3]
        self.mlp = MLP(**config)

    def forward(self, x):
        prob = self.mlp(x)

        return prob


class Value(nn.Module):
    def __init__(self, config):
        super(Value, self).__init__()

        hs1 = config.input_size * 10
        hs3 = 5
        hs2 = int(np.round(np.sqrt(hs1 * hs3)))
        config.hidden_size = [hs1, hs2, hs3]
        config.output_size = 1
        self.mlp = MLP(**config)

    def forward(self, x):
        value = self.mlp(x)

        return value


class Model:
    def __init__(self, config, verbose=True):

        self.config = config

        count_parameters = lambda m: sum(p.numel() for p in m.parameters())

        self.policy = Policy(config.policy)
        self.policy.cuda()
        if verbose:
            print(self.policy)
            print('# of parameters: {}\n'.format(count_parameters(self.policy)))

        self.value = Value(config.value)
        self.value.cuda()
        if verbose:
            print(self.value)
            print('# of parameters: {}\n'.format(count_parameters(self.value)))

        # self.optimizer = getattr(optim, config.optimizer.algorithm)([{
            # 'params': self.policy.parameters(),
            # 'lr': config.optimizer.policy_learning_rate
        # }, {
            # 'params': self.value.parameters(),
            # 'lr': config.optimizer.value_learning_rate
        # }])
        self.policy_optimizer = getattr(optim, config.policy_optimizer.algorithm)(
                self.policy.parameters(), lr=config.policy_optimizer.learning_rate)

        self.value_optimizer = getattr(optim, config.value_optimizer.algorithm)(
                self.value.parameters(), lr=config.value_optimizer.learning_rate)

        if config.ckpt_path:
            self.load_state(config.ckpt_path)

    def set_train(self):
        self.policy.train()
        self.value.train()

    def set_eval(self):
        self.policy.eval()
        self.value.eval()

    # def select_action(self, obs):
        # mean, std = self.policy(obs)
        # act = torch.normal(mean, std)

        # val = self.value(obs)

        # return mean, std, act, val

    def select_action(self, obs):
        prob = self.policy(obs)
        act = prob.multinomial(1).data[0, 0]
        val = self.value(obs)

        return prob, act, val

    def get_policy_state(self):
        return self.policy.state_dict()

    def set_policy_state(self, state):
        self.policy.load_state_dict(state)

    def load_state(self, ckpt_path):
        print('Loading model state from {}'.format(ckpt_path))
        ckpt = torch.load(ckpt_path)
        print(ckpt['info'])
        self.policy.load_state_dict(ckpt['policy_state'])
        self.value.load_state_dict(ckpt['value_state'])
        self.policy_optimizer.load_state_dict(ckpt['policy_optimizer_state'])
        self.value_optimizer.load_state_dict(ckpt['value_optimizer_state'])

    def save_state(self, info, ckpt_path, is_best):
        torch.save({
            'info': info,
            'policy_state': self.policy.state_dict(),
            'value_state': self.value.state_dict(),
            'policy_optimizer_state': self.policy_optimizer.state_dict(),
            'value_optimizer_state': self.value_optimizer.state_dict()
        }, ckpt_path)
        if is_best:
            best_ckpt_path = os.path.join(os.path.dirname(ckpt_path),
                    'best_model.ckpt')
            shutil.copy(ckpt_path, best_ckpt_path)
