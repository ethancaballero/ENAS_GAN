import os, sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
import torch.multiprocessing as mp
import torch.optim as optim
from torch import autograd
import random
import rnn_controller
from arguments import get_args


def rollout_ppo(args, actions, net, controller):
    net.zero_grad()
    action_log_probs_list = []
    dist_entropy_list = []
    h_state = None
    for _i in range(net.required_code_length()):
        get_value = True if _i == net.required_code_length() - 1 else False
        if _i:
            inputs = actions[_i-1]
        else:
            inputs = torch.zeros(actions[0].size()).type(torch.LongTensor)
        value, action_log_probs, dist_entropy, h_state = controller.evaluate_actions(V(inputs), h_state, V(actions[_i]),
                                                                                     get_value=get_value)
        action_log_probs_list.append(action_log_probs)
        dist_entropy_list.append(dist_entropy)

    return value, torch.stack(action_log_probs_list), torch.stack(dist_entropy_list)
