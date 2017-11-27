import torch
import numpy as np

class RolloutStorage(object):
    #def __init__(self, num_steps, num_processes, obs_shape, action_space, state_size):
    def __init__(self, num_steps, num_processes):
        #self.logprobs = torch.zeros(num_steps, num_processes)
        #self.ents = torch.zeros(num_steps, num_processes)
        self.logprobs = None
        self.ents = None
        #self.rewards = torch.zeros(num_processes)
        self.rewards_GAN = torch.zeros(num_processes)
        self.rewards_INCEPT = torch.zeros(num_processes)
        #self.values = torch.zeros(num_processes)
        self.values = None

        #ewma baseline based off of https://github.com/hans/thinstack-rl/blob/master/reinforce.py#L4
        #self.avg_reward = torch.zeros(1)
        self.avg_reward_GAN = torch.zeros(1)
        self.avg_reward_INCEPT = torch.zeros(1)

    def cuda(self):
        self.logprobs = self.logprobs.cuda()
        self.ents = self.ents.cuda()
        self.rewards = self.rewards.cuda()
        self.values = self.values.cuda()
        #self.avg_reward = self.avg_reward.cuda()
        self.avg_reward_GAN = self.avg_reward_GAN.cuda()
        self.avg_reward_INCEPT = self.avg_reward_INCEPT.cuda()

    def insert(self, logprob, ent, value):
        self.logprobs = logprob
        self.ents = ent
        self.values = value

    def insert_reward_GAN(self, process, reward_GAN):
        self.rewards_GAN[process] = reward_GAN[0]

    def insert_reward_INCEPT(self, process, reward_INCEPT):
        self.rewards_INCEPT[process] = reward_INCEPT[0]

    def update_avg_reward_GAN(self, tau=.84):
        self.avg_reward_GAN = tau * self.avg_reward_GAN + (1. - tau) * torch.mean(self.rewards_GAN)

    def update_avg_reward_INCEPT(self, tau=.84):
        self.avg_reward_INCEPT = tau * self.avg_reward_INCEPT + (1. - tau) * torch.mean(self.rewards_INCEPT)

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                    gamma * self.masks[step + 1] + self.rewards[step]
