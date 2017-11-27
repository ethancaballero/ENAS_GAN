import sys
import math
import random as r
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn import Parameter as P
from torch.autograd import Variable as V
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

def orthogonal(tensor, gain=1):
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = torch.Tensor(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    q, r = torch.qr(flattened)
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph.expand_as(q)

    if rows < cols:
        q.t_()

    tensor.view_as(q).copy_(q)
    tensor.mul_(gain)
    return tensor

#not using this module anymore
class LinearOptionalTranpose(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = A_transpose*x + b`
    
    modification of linear that allows transposing self.weight
    so that tying encoder and decoder can be more easily applied
    as in https://arxiv.org/abs/1611.01462
    while still permitting gumbel softmax trick to be used
    """

    def __init__(self, in_features, out_features, bias=False):
        super(LinearOptionalTranpose, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        #self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, transpose=False):
        if transpose==True:
            return F.linear(input, self.weight.t(), self.bias)
        else:
            return F.linear(input, self.weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

class Categorical(nn.Module):
    def __init__(self, args, decoder):
        super(Categorical, self).__init__()
        self.args = args
        self.linear = decoder

    def forward(self, x):
        x = self.linear(x)
        return x

    def sample(self, x, deterministic):
        x = self(x)
        probs = F.softmax(x)
        if deterministic is False:
            action = probs.multinomial()
        else:
            action = probs.max(1)[1]
        return action

    def logprobs_and_entropy(self, x, actions):
        x = self(x)

        log_probs = F.log_softmax(x)
        probs = F.softmax(x)

        action_log_probs = log_probs.gather(1, actions)

        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return action_log_probs, dist_entropy

class Controller(nn.Module):
    def __init__(self, args, dim, vocab_size, layers=2):
        super(Controller, self).__init__()
        self.args = args
        self.dim = dim
        self.vocab_size = vocab_size
        self.layers = layers

        #self.linear_enc_dec = LinearOptionalTranpose(self.vocab_size, self.dim, False)
        self.enc = nn.Embedding(self.vocab_size, self.dim)
        self.dec = nn.Linear(self.dim, self.vocab_size)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        self.dec.weight = self.enc.weight

        #self.lstm = nn.LSTMCell(self.dim, self.dim)
        self.lstm = nn.LSTM(self.dim, self.dim, self.layers)
        self.dist = Categorical(self.args, self.dec)
        self.critic_linear = nn.Linear(self.dim, 1)

        self.train()
        self.reset_parameters()

    def reset_parameters(self):
        #print(self.lstm.__dict__)
        if hasattr(self, 'lstm'):
            #TODO: it's hardcoded to 2 layers for now
            if self.layers == 2:
                orthogonal(self.lstm.weight_ih_l0.data)
                orthogonal(self.lstm.weight_hh_l0.data)
                self.lstm.bias_ih_l0.data.fill_(0)
                self.lstm.bias_hh_l0.data.fill_(0)
                orthogonal(self.lstm.weight_ih_l1.data)
                orthogonal(self.lstm.weight_hh_l1.data)
                self.lstm.bias_ih_l1.data.fill_(0)
                self.lstm.bias_hh_l1.data.fill_(0)
            else:
                error

        orthogonal(self.critic_linear.weight.data)
        self.critic_linear.bias.data.fill_(0)

    def forward(self, inputs, states=None, get_value=False):
        x = self.enc(inputs)
        x = x.permute(1,0,2)

        x, h_states = self.lstm(x, states)

        '''TODO: make sure -1 is the top lstm layer'''
        '''This makes me think it is: https://github.com/salesforce/awd-lstm-lm/blob/master/model.py#L85'''
        if get_value:
            return self.critic_linear(x[0]), x[0], h_states
        else:
            return None, x[0], h_states

    '''TODO IMMEDIATE: SHOULD THE TEMPERATURE BE SET TO 1 DURING INFERENCE'''

    def act(self, inputs, states, deterministic=False, get_value=False):
        value, x, states = self(inputs, states, get_value=get_value)
        x = self.args.tanh_constant * torch.tanh(x/self.args.temperature)
        action = self.dist.sample(x, deterministic=deterministic)
        return value, action, states

    def evaluate_actions(self, inputs, states, actions, get_value=False):
        value, x, states = self(inputs, states, get_value=get_value)
        x = self.args.tanh_constant * torch.tanh(x/self.args.temperature)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, actions)
        return value, action_log_probs, dist_entropy

    def act_and_evaluate(self, inputs, states, deterministic=False, get_value=False):
        value, x, states = self(inputs, states, get_value=get_value)
        x = self.args.tanh_constant * torch.tanh(x/self.args.temperature)
        #action = self.dist.sample(x, deterministic=deterministic)
        actions = self.dist.sample(x, deterministic=deterministic)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, V(actions.data))
        #return value, action, states, action_log_probs, dist_entropy
        return value, actions, states, action_log_probs, dist_entropy










