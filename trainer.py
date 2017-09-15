import os
from tqdm import tqdm
from IPython import embed

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils import MemoryDataset, collate_fn, normal, normalize, ZFilter
from model import Model
from plotter import Plotter

class Trainer:
    def __init__(self, config, env, model, model_config):
        self.config = config
        self.env = env
        self.model = model
        self.model_config = model_config

        self.obs_zfilter = ZFilter(env.observation_space.shape, clip=None)
        self.r_zfilter = ZFilter((1), demean=False, clip=None)

        self.stats = {
            'reward': []
        }
        self.best_eval_score = float('-INF')

        window_config_list = [{
            'num_traces': 1,
            'opts': {
                'title': '[Train] Reward',
                'xlabel': 'Iteration',
                'ylabel': 'Reward',
                'width': 900,
                'height': 400,
                'margintop': 100
            }
        }, {
            'num_traces': 1,
            'opts': {
                'title': '[Eval] Reward',
                'xlabel': 'Iteration',
                'ylabel': 'Reward',
                'width': 900,
                'height': 400,
                'margintop': 100
            }
        }]

        self.plotter = Plotter(self.config.experiment, window_config_list)

        train_dir = os.path.join(config.ckpt_dir, config.experiment)
        if not os.path.isdir(train_dir):
            os.makedirs(train_dir)

    def start(self):
        for rnd in range(self.config.num_rounds):
            self.train(rnd)
            self.eval(rnd)
            self.plotter.save()

    def train(self, rnd):
        self.model.set_train()
        for iteration in range(self.config.num_train_iterations):
            global_iteration = rnd * self.config.num_train_iterations \
                    + iteration + 1
            desc = '[Train | Iter {:3d}] Collecting rollout'.format(
                    global_iteration)
            t = tqdm(range(self.config.num_train_episodes), desc=desc)
            memory = MemoryDataset()
            for episode in t:
                data = self.run_episode()
                memory.append(**data)

            print('obs | mean: {}, std: {}'.format(
                self.obs_zfilter.rs.mean, self.obs_zfilter.rs.std))
            # print('r | mean: {}, std: {}'.format(
                # self.r_zfilter.rs.mean, self.r_zfilter.rs.std))

            data_loader = DataLoader(memory,
                    batch_size = self.config.batch_size, shuffle=True,
                    collate_fn=collate_fn)
            old_model = Model(self.model_config, verbose=False)
            old_model.set_policy_state(self.model.get_policy_state())
            old_model.set_train()
            for epoch in range(self.config.num_train_epochs):
                desc = '[Train | Iter {:3d}] Update epoch {:2d}'.format(
                        global_iteration, epoch)
                t = tqdm(data_loader, desc=desc)
                for batch in t:
                    observation = Variable(batch['observation'])
                    action = batch['action']
                    action_index = (range(len(action)), action)
                    reward = Variable(batch['reward'])
                    advantage = Variable(batch['advantage'])
                    advantage = normalize(advantage)

                    # old_mean, old_std, _, _ = old_model.select_action(observation)
                    # old_prob = normal(old_mean, old_std, action)

                    # mean, std, _, value = self.model.select_action(observation)
                    # prob = normal(mean, std, action)

                    old_prob, _, _ = old_model.select_action(observation)
                    old_prob = old_prob[action_index].view(-1, 1)

                    prob, _, value = self.model.select_action(observation)
                    prob = prob[action_index].view(-1, 1)

                    ratio = prob / (1e-16 + old_prob)
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - self.model_config.epsilon,
                            1 + self.model_config.epsilon) * advantage
                    clip_loss = -torch.mean(torch.min(surr1, surr2))

                    entropy = prob * torch.log(prob + 1e-16)
                    entropy_loss = self.model_config.beta * torch.mean(entropy)

                    # old_model.set_policy_state(self.model.get_policy_state())

                    policy_loss = clip_loss + entropy_loss
                    self.model.policy_optimizer.zero_grad()
                    policy_loss.backward()
                    self.model.policy_optimizer.step()

                    value_loss = torch.mean((value - reward) ** 2)
                    self.model.value_optimizer.zero_grad()
                    value_loss.backward()
                    self.model.value_optimizer.step()
            memory.clear()

            if (iteration + 1) % self.config.log_interval == 0:
                self.plot(global_iteration, 'train')

    def eval(self, rnd):
        self.model.set_eval()
        global_iteration = (rnd + 1) * self.config.num_train_iterations
        desc = '[Eval  | Iter {:3d}] Running evaluation'.format(
                global_iteration)
        t = tqdm(range(self.config.num_eval_episodes), desc=desc)
        for episode in t:
            _ = self.run_episode()

        eval_score = np.mean(self.stats['reward'])
        is_best = False
        if eval_score > self.best_eval_score:
            is_best = True
            self.best_eval_score = eval_score

        info = {
            'iteration': global_iteration,
            'eval_score': eval_score
        }

        ckpt_path = os.path.join(self.config.ckpt_dir, self.config.experiment,
                'model-{}.ckpt'.format(rnd))
        self.model.save_state(info, ckpt_path, is_best)

        self.plot(global_iteration, 'eval')

    def run_episode(self):
        observations = [] # cuda tensor of shape (1,) + env.observation_space.shape
        # actions = []      # cuda tensor of shape (1,) + env.action_space.shape
        actions = []      # int
        values = []       # cuda tensor of shape (1, 1)
        rewards = []      # float
        self.stats['reward'].append([])

        done = False
        obs = self.env.reset()
        while not done:
            obs = self.obs_zfilter(obs)
            obs = torch.from_numpy(obs).float().unsqueeze(0).cuda()
            observations.append(obs)

            _, act, val = self.model.select_action(Variable(obs, volatile=True))
            actions.append(act)
            values.append(val.data)
            # act = act.data.squeeze().cpu().numpy()

            obs, r, done, _ = self.env.step(act)
            self.stats['reward'][-1].append(r)
            # r = self.r_zfilter(np.array([r]))[0]
            rewards.append(r)

        self.stats['reward'][-1] = sum(self.stats['reward'][-1])

        obs = self.obs_zfilter(obs)
        obs = torch.from_numpy(obs).float().unsqueeze(0).cuda()
        _, _, val = self.model.select_action(Variable(obs, volatile=True))
        values.append(val.data)

        R = torch.zeros(1, 1).cuda()
        A = torch.zeros(1, 1).cuda()
        acc_rewards = []
        advantages = []
        for i in reversed(range(len(rewards))):
            R = rewards[i] + self.model_config.gamma * R
            acc_rewards.insert(0, R)

            delta = rewards[i] + self.model_config.gamma * values[i + 1] \
                    - values[i]
            A = delta + self.model_config.gamma * self.model_config.lmbda * A
            advantages.insert(0, A)

        return {
            'observations': observations,
            'actions': actions,
            'rewards': acc_rewards,
            'advantages': advantages
        }

    def plot(self, global_iteration, mode):
        upper = lambda s: s[0].upper() + s[1:]
        title_prefix = '[{}]'.format(upper(mode))

        reward = np.mean(self.stats['reward'])
        stats_list = [{
            'title': '{} Reward'.format(title_prefix),
            'X': global_iteration,
            'Y': reward
        }]
        self.plotter.update(stats_list)

        prefix = title_prefix + (' ' if mode == 'eval' else '')
        print('{} Iteration {:5d} | Reward {:.5f}\n'.format(
                prefix, global_iteration, reward))

        self.clear_stats()

    def clear_stats(self):
        self.stats = {
            'reward': []
        }
