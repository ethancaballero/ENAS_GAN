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
from arguments import get_args

args = get_args()

# The unit of adjustable channels, S, is equal to max_ch/R, where R is set to be 8 by default.

R = 4
M = 5  # The number of possible masks plus one spot reserved for skip connection


# By default, 1x1 conv, 3x3 conv, max_pool and avg_pool are used for mask
# If the kind of masks to be used is modified, one needs to modify Layer()
# as well as in_ch_stab()


class Agent(nn.Module):
    def __init__(self, resolution, layers_per_block):
        super(Agent, self).__init__()
        self.num_blocks = int(math.log((resolution // 4), 2)) + 1  # We set minimum resolution to be 4
        self.layers_per_block = layers_per_block
        self.blocks = nn.ModuleList()
        self.transes = nn.ModuleList()

    def nf(self, stage):  # nf(0):=latent_size
        # nf(stage) counts the "base" channel number of the block of the stage,
        # which is equal to the maximum possible channel number of a single branch of a generic layer
        # This means that 4*(base channel number) is the maximum possible channel num per layer, and
        # that random search gives 2*(base channel number) as the average channel num per layer, which
        # is a good estimate of how many channels per layer there are in a given block
        # If num_blocks=4, the following plan goes like, for G, 512(latent)-> 256*2 (1st layer)-> 128*2-> 64*2-> 32*2
        # where we listed 2*(base channel number)'s
        return 512 // (2 ** stage)

    def _make_trans(self, type):
        return [Scale(type)]

    def precise_layers_per_block(self, i):
        if i == 0:
            return self.layers_per_block - 1  # smallest block has one less layers
        elif i == -1:  # for trans blocks
            return 1
        else:
            return self.layers_per_block

    def _make_block(self, in_ch, prev_in_ch, i, type):
        return [Block(layers_per_block=self.precise_layers_per_block(i), in_ch=in_ch, prev_in_ch=prev_in_ch,
                      type=type)]

    def vocab_size(self):
        return 5

    def forward(self, x, code):
        return NotImplementedError


class G(Agent):
    def __init__(self, resolution, layers_per_block=5):
        super(G, self).__init__(resolution, layers_per_block)

        # stemconv: 1x1 output to 4x4 output
        self.stem_conv = Conv(self.nf(0), self.nf(1), 4, stride=4, padding=0, deconv=True)

        # assert self.num_blocks == 4
        # Note that progGAN used 512(latent),512,512,512,256 for CIFAR
        # The default setting is 512(latent)-> 512 (1st layer)-> 256-> 128-> 64
        self.blocks += self._make_block(self.nf(1), self.nf(1), 0, type='G')
        for i in range(1, self.num_blocks):
            self.blocks += self._make_block(self.nf(i + 1), self.nf(i), 1, type='G')
        for i in range(self.num_blocks - 1):
            self.transes += [Layer(self.nf(i + 1), self.nf(i + 1), self.nf(i + 2))]

        self.end_conv = Conv(self.nf(self.num_blocks), 3, 1, padding=0, linear=True, norm_used=False)

    def required_code_length(self):  # The length of code to be used
        return 36

    def forward(self, x, code):
        '''
        :param x: latent noise (batch_size x dimension)
        :param code: Python list consisting of G.required_code_length() elements, each of which is an integer from 1 to 2^R
        :return: x: Batch of generated images (batch_size x 3 x resolution x resolution)
        :return: params: VERY rough estimate of the number of parameters of G
        '''
        c = ""
        for i in range(4):
            tmp = str(bin(code[i] % 4))[2:]
            tmp += "0" * (2 - len(tmp))
            c += tmp

        params = 0  # VERY rough estimate of number of parameters
        code_normal = code[4:20]
        code_trans = code[20:36]

        if args.cuda:
            zeros = V(torch.zeros(x.size(0), self.nf(0) // R).cuda(),
                      volatile=not self.training, requires_grad=False)
        else:
            zeros = V(torch.zeros(x.size(0), self.nf(0) // R),
                      volatile=not self.training, requires_grad=False)
        list = []
        for i in range(R):
            if c[i] == '1':
                list.append(x[:, (self.nf(0) // R) * i:(self.nf(0) // R) * (i + 1)])
            else:
                list.append(zeros)
        x = torch.cat(list, 1)
        x = x.view(x.size(0), -1, 1, 1)

        x = pn(x)
        x = self.stem_conv(x)
        params += 4 * 4 * self.nf(0) * self.nf(1)
        for i in range(self.num_blocks):
            if i == 0:
                h, x, params_tmp = self.blocks[i](x, x, code_normal)
            else:
                h, x, params_tmp = self.blocks[i](h, x, code_normal)
            params += params_tmp
            if i != self.num_blocks - 1:
                tmp = h
                h, params_tmp = self.transes[i](h, x, code_trans)
                x = tmp
                del tmp
                params += params_tmp
        h = self.end_conv(h)
        return h, params


class D(Agent):
    def __init__(self, resolution, layers_per_block=5):
        super(D, self).__init__(resolution, layers_per_block)

        self.stem_conv = Conv(3, self.nf(self.num_blocks), 1, padding=0, norm_used=False)

        self.blocks += self._make_block(self.nf(self.num_blocks), self.nf(self.num_blocks), 1, type='D')
        for i in range(self.num_blocks - 1, 1, -1):
            self.blocks += self._make_block(self.nf(i), self.nf(i + 1), 1, type='D')
        self.blocks += self._make_block(self.nf(1), self.nf(2), 0, type='D')  # smallest block has one less layer
        for i in range(self.num_blocks, 1, -1):
            self.transes += [Layer(self.nf(i), self.nf(i), self.nf(i - 1))]

        self.stddev = MinibatchStatConcatLayer()

        self.end_conv = Conv(self.nf(1) + 1, self.nf(0), 4, padding=0, stride=4, norm_used=False)
        self.fc = Linear(self.nf(0), 1)

    def required_code_length(self):  # The length of code to be used
        return 32

    def forward(self, x, code):
        '''
        :param x: Batch of generated images (batch_size x 3 x resolution x resolution)
        :param code: Python list consisting of D.required_code_length() elements, each of which is an integer from 1 to 2^R
        :return: x: Scalar output (batch_size x 1)
        :return: params: VERY rough estimate of the number of parameters of D
        '''
        params = 0  # VERY rough estimate of number of parameters
        code_normal = code[0:16]
        code_trans = code[16:32]

        x = self.stem_conv(x)
        for i in range(self.num_blocks):
            if i == 0:
                h, x, params_tmp = self.blocks[i](x, x, code_normal)
            else:
                h, x, params_tmp = self.blocks[i](h, x, code_normal)
            params += params_tmp
            if i != self.num_blocks - 1:
                tmp = h
                h, params_tmp = self.transes[i](h, x, code_trans)
                x = tmp
                del tmp
                params += params_tmp
        h = self.stddev(h)
        h = self.end_conv(h)
        params += 4 * 4 * self.nf(1) * self.nf(0)
        h = self.fc(h.view(-1, h.size(1)))
        return h, params


class Layer(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch):
        super(Layer, self).__init__()
        self.in_ch = [in_ch1, in_ch2]
        self.out_ch = out_ch
        if self.in_ch[0] != self.in_ch[1]:
            self.adjust = Factorized_adjustment(in_ch1, in_ch2)

        self.conv1 = nn.ModuleList()
        for i in range(2):
            for j in range(4):
                self.conv1 += self._make_layer(self.in_ch[0], kernel=1, padding=0,
                                               stride=1 if self.in_ch[0] >= self.out_ch else 2)
        for i in range(6):
            self.conv1 += self._make_layer(out_ch // 4, kernel=1, padding=0)
        for k in range(3):
            for i in range(2):
                for j in range(4):
                    self.conv1 += self._make_layer(self.in_ch[0], kernel=1, padding=0,
                                                   stride=1 if self.in_ch[0] >= self.out_ch else 2)

        self.conv3 = nn.ModuleList()
        for i in range(2):
            for j in range(4):
                self.conv3 += self._make_layer(self.in_ch[0], kernel=3, padding=1,
                                               stride=2 if self.in_ch[0] < self.out_ch else 1)
        for i in range(6):
            self.conv3 += self._make_layer(out_ch // 4, kernel=3, padding=1)

    def _make_layer(self, in_ch, kernel, padding, stride=1, deconv=False):
        return [Conv(in_ch, self.out_ch // 4, kernel, padding=padding, stride=stride, deconv=deconv)]

    def forward(self, h1, h2, code):
        print(h1.size())

        c = [[[] for j in range(4)] for i in range(5)]

        for i in range(4):
            tmp = [code[4 * i] % 5, code[4 * i + 1] % (i + 2), code[4 * i + 2] % 5, code[4 * i + 3] % (i + 2)]
            if tmp[1] == tmp[3] and tmp[0] == tmp[2]:
                tmp[3] = (code[4 * i + 3] + 1) % (i + 2)
            c[tmp[1]][i].append(tmp[0])
            c[tmp[3]][i].append(tmp[2])

        params = 0
        scale = h1.size(2)
        if self.in_ch[0] > self.out_ch:
            scale = scale * 2
        elif self.in_ch[0] < self.out_ch:
            scale = scale // 2
        if args.cuda:
            zeros_out = V(torch.zeros(h1.size(0), self.out_ch // 4, scale, scale).cuda(),
                          volatile=not self.training, requires_grad=False)
        else:
            zeros_out = V(torch.zeros(h1.size(0), self.out_ch // 4, scale, scale), volatile=not self.training,
                          requires_grad=False)
        h = [h1, h2]
        del h1, h2
        if self.in_ch[0] != self.in_ch[1]:
            h[1], p = self.adjust(h[1], params)
            params += p
        sum = [zeros_out] * 4
        for i in range(2):
            for j in range(4):
                for k in c[i][j]:
                    if k == 0:
                        if self.in_ch[0] > self.out_ch:
                            sum[j] = sum[j] + self.conv1[4 * i + j](F.upsample(h[i], scale_factor=2))
                        else:
                            sum[j] = sum[j] + self.conv1[4 * i + j](h[i])
                        params += self.in_ch[0] * self.out_ch // 4
                    if k == 1:
                        if self.in_ch[0] > self.out_ch:
                            sum[j] = sum[j] + self.conv3[4 * i + j](F.upsample(h[i], scale_factor=2))
                        else:
                            sum[j] = sum[j] + self.conv3[4 * i + j](h[i])
                        params += 3 * 3 * self.in_ch[0] * self.out_ch // 4
                    if k == 2:
                        if self.out_ch >= self.in_ch[0]:
                            sum[j] = sum[j] + self.conv1[14 + 4 * i + j](F.avg_pool2d(h[i], 3, padding=1, stride=1))
                        else:
                            sum[j] = sum[j] + self.conv1[14 + 4 * i + j](
                                F.upsample(F.avg_pool2d(h[i], 3, 1, 1), scale_factor=2))
                        params += self.in_ch[0] * self.out_ch // 4
                    if k == 3:
                        if self.out_ch >= self.in_ch[0]:
                            sum[j] = sum[j] + self.conv1[22 + 4 * i + j](F.max_pool2d(h[i], 3, 1, 1))
                        else:
                            sum[j] = sum[j] + self.conv1[22 + 4 * i + j](
                                F.upsample(F.max_pool2d(h[i], 3, 1, 1), scale_factor=2))
                        params += self.in_ch[0] * self.out_ch // 4
                    if k == 4:
                        if self.in_ch[0] > self.out_ch:
                            sum[j] = sum[j] + self.conv1[30 + 4 * i + j](F.upsample(h[i], scale_factor=2))
                        else:
                            sum[j] = sum[j] + self.conv1[30 + 4 * i + j](h[i])
                        params += self.in_ch[0] * self.out_ch // 4
        del h
        for i in range(6):
            if i == 0:
                j = 0
                k = 1
            elif i == 1:
                j = 0
                k = 2
            elif i == 2:
                j = 0
                k = 3
            elif i == 3:
                j = 1
                k = 2
            elif i == 4:
                j = 1
                k = 3
            else:
                j = 2
                k = 3
            for l in c[j][k]:
                if l == 0:
                    sum[k] = sum[k] + self.conv1[8 + i](sum[j])
                params += self.out_ch * self.out_ch // 16
                if l == 1:
                    sum[k] = sum[k] + self.conv3[8 + i](sum[j])
                params += 3 * 3 * self.out_ch * self.out_ch // 16
                if l == 2:
                    sum[k] = sum[k] + F.leaky_relu(F.avg_pool2d(sum[j], 3, 1, 1),
                                           negative_slope=0.2)
                if l == 3:
                    sum[k] = sum[k] + F.leaky_relu(F.max_pool2d(sum[j], 3, 1, 1),
                                           negative_slope=0.2)
                if l == 4:  # identity
                    sum[k] = sum[k]
        output = torch.cat(sum, 1)
        del sum
        return output, params


# Note that channel size changes differently for G and D for the sake of symmetry
class Block(nn.Module):
    def __init__(self, layers_per_block, in_ch, prev_in_ch, type):
        super(Block, self).__init__()
        self.layers_per_block = layers_per_block
        self.in_ch = in_ch
        self.prev_in_ch = prev_in_ch
        self.type = type  # 'D' or 'G'
        self.layers = nn.ModuleList()
        self.layers += [Layer(in_ch, prev_in_ch, in_ch)]
        for i in range(self.layers_per_block - 1):
            self.layers += [Layer(in_ch, in_ch, in_ch)]

    def forward(self, h1, h2, code):

        params = 0
        for i in range(self.layers_per_block):
            x, p = self.layers[i](h1, h2, code)
            h2 = h1
            h1 = x
            params += p

        return h1, h2, params


def pn(x):  # Pixel_norm
    return x / torch.sqrt(torch.mean(x * x, dim=1, keepdim=True) + 1e-8)


# -----------------
# Custom linear, conv, convtranspose layers to accommodate weight scaling and pixel norm
# Beware that norm_used=False for D, but Conv sets norm_used=True by default

class Linear(nn.Linear):
    def __init__(self, in_units, out_units):
        super(Linear, self).__init__(in_units, out_units)

    def forward(self, x):
        scale = torch.mean(self.weight.data ** 2) ** 0.5
        # x = F.linear(x, self.weight / scale.view(1, 1).expand_as(self.weight), self.bias)
        x = F.linear(x, self.weight / scale, self.bias)
        x = x * scale
        return x + torch.unsqueeze(self.bias, 0)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 padding, stride=1, deconv=False, linear=False, norm_used=True):
        super(Conv, self).__init__()
        if deconv:
            self.conv = Conv_transpose(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding, linear=linear,
                norm_used=norm_used)
        else:
            self.conv = Conv_regular(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding, linear=linear,
                norm_used=norm_used)

    def forward(self, x):
        return self.conv(x)


class Conv_transpose(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, linear=False, norm_used=True):
        super(Conv_transpose, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = None
        self.linear = linear
        self.norm_used = norm_used

    def forward(self, x):
        scale = torch.mean(self.weight.data ** 2) ** 0.5
        x = F.conv_transpose2d(x, self.weight / scale, stride=self.stride,
                               padding=self.padding)
        # x = F.conv_transpose2d(x, self.weight, stride=self.stride, padding=self.padding)

        x = x * scale
        if not self.linear:
            x = F.leaky_relu(x, negative_slope=0.2)
        if self.norm_used:
            x = pn(x)
        return x


# Basically the same as Conv_transpose except that it's nn.Conv2d counterpart...
# I'm not sure whether I can merge them, since inheritance would be inconsistent if merged
class Conv_regular(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, linear=False, norm_used=True):
        super(Conv_regular, self).__init__(in_channels, out_channels, kernel_size,
                                           stride=stride, padding=padding)
        self.bias = None
        self.linear = linear
        self.norm_used = norm_used

    def forward(self, x):
        scale = torch.mean(self.weight.data ** 2) ** 0.5
        x = F.conv2d(x, self.weight / scale, stride=self.stride,
                     padding=self.padding)
        # x = F.conv2d(x, self.weight, stride=self.stride, padding=self.padding)
        x = x * scale
        if not self.linear:
            x = F.leaky_relu(x, negative_slope=0.2)
        if self.norm_used:
            x = pn(x)
        return x


# This is used for transition
class Scale(nn.Module):
    def __init__(self, type):
        super(Scale, self).__init__()
        self.type = type

    def forward(self, x):
        if self.type == 'G':
            return F.upsample(x, scale_factor=2)
        else:  # 'down'
            return F.avg_pool2d(x, kernel_size=2, stride=2, count_include_pad=False)


# --------------------------

class MinibatchStatConcatLayer(nn.Module):
    def __init__(self):
        super(MinibatchStatConcatLayer, self).__init__()

    def square(self, x): return torch.mul(x, x)

    def forward(self, input):
        reps = list(input.size())
        vals = torch.sqrt(torch.mean(self.square(input - torch.mean(input, 0, keepdim=True)), 0, keepdim=True) + 1.0e-8)
        vals = torch.mean(vals.view(-1), 0, keepdim=True)
        reps[1] = 1
        vals = vals.repeat(*reps)
        return torch.cat([input, vals], 1)


class Factorized_adjustment(nn.Module):
    def __init__(self, in_ch, prev_in_ch):
        super(Factorized_adjustment, self).__init__()
        self.in_ch = in_ch
        self.prev_in_ch = prev_in_ch
        self.conv = Conv(prev_in_ch, in_ch, 1, padding=0)

    def forward(self, x, params):
        params += self.in_ch * self.prev_in_ch
        if self.prev_in_ch > self.in_ch:
            return self.conv(F.upsample(x, scale_factor=2)), params
        else:
            return self.conv(F.avg_pool2d(x, 3, padding=1, stride=2)), params


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        init.kaiming_normal(m.weight.data)
    elif isinstance(m, nn.Linear):
        init.kaiming_normal(m.weight.data, a=1)
        m.bias.data.zero_()


# For testing:

g = G(32)
d = D(32)
g.apply(weights_init)
d.apply(weights_init)
#d.apply(weights_init)
print(d.required_code_length())
coded, codeg = [], []
for i in range(g.required_code_length()):
    codeg.append(r.randint(0,4))
for i in range(d.required_code_length()):
    coded.append(r.randint(0,4))
from functools import reduce

y,_=g(V(torch.randn(2,512),volatile=True),codeg)
x=d(y,coded)
print(y,x)

# print(d(x,code))
# gp = sum([reduce(lambda x, y: x * y, p.size()) for p in g.parameters()])
# dp = sum([reduce(lambda x, y: x * y, p.size()) for p in d.parameters()])

# print(gp)
# print(dp)

# print(g(V(torch.randn(2,512)),code))