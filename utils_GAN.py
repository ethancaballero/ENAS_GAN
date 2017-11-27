import os, sys
sys.path.append(os.getcwd())

import time
import tflib as lib
import tflib.save_images
import tflib.mnist
import tflib.cifar10
import tflib.plot
import tflib.inception_score

import numpy as np


import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim

#from train import DIM
use_cuda = torch.cuda.is_available()

# For generating samples
def generate_image(frame, netG):
    fixed_noise = torch.randn(DIM, DIM)
    if use_cuda:
        fixed_noise = fixed_noise.cuda(gpu)
    noisev = autograd.Variable(fixed_noise, volatile=True)
    samples = netG(noisev)
    samples = samples.view(-1, 3, 32, 32)
    samples = samples.mul(0.5).add(0.5)
    samples = samples.cpu().data.numpy()

    lib.save_images.save_images(samples, './tmp/cifar10/samples_{}.jpg'.format(frame))

# For calculating inception score
def get_inception_score(G, ):
    all_samples = []
    for i in xrange(10):
        samples_100 = torch.randn(100, DIM)
        if use_cuda:
            samples_100 = samples_100.cuda(gpu)
        samples_100 = autograd.Variable(samples_100, volatile=True)
        all_samples.append(G(samples_100).cpu().data.numpy())

    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = np.multiply(np.add(np.multiply(all_samples, 0.5), 0.5), 255).astype('int32')
    all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    return lib.inception_score.get_inception_score(list(all_samples))

def get_inception_score_m(G, code):
    all_samples = []

    samples_100 = torch.randn(100, DIM)
    if use_cuda:
        samples_100 = samples_100.cuda(gpu)
    samples_100 = autograd.Variable(samples_100, volatile=True)
    all_samples.append(G(samples_100, code).cpu().data.numpy())

    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = np.multiply(np.add(np.multiply(all_samples, 0.5), 0.5), 255).astype('int32')
    all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    return lib.inception_score.get_inception_score(list(all_samples))

preprocess = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])

def get_mmd_score(G, code, data):
    # data: tensor of randomly sampled real images (batch x 3 x res x res)
    # 'batch' should be roughly about 100
    batch = data[0]
    all_samples = []

    samples = torch.randn(batch, DIM)
    if use_cuda:
        samples = samples.cuda(gpu)
        data = data.cuda(gpu)
    samples = autograd.Variable(samples, volatile=True)
    data = autograd.Variable(data, volatile=True)
    # F is the inception-v3 or NASNet-A_Mobile, the latter of which can be found here:
    # https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet
    x = torch.mean(F(G(samples, code))-F(data), 0)
    return torch.sqrt(torch.sum(x*x))