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

# The unit of adjustable channels, S, is equal to max_ch/R, where R is set to be 8 by default.

R=8
M=5 # The number of possible masks plus one spot reserved for skip connection
    # By default, 1x1 conv, 3x3 conv, max_pool and avg_pool are used for mask
    # If the kind of masks to be used is modified, one needs to modify Layer()
    # as well as in_ch_stab()


class Agent(nn.Module):
    def __init__(self, resolution, layers_per_block):
        super(Agent, self).__init__()
        self.num_blocks = int(math.log((resolution//4),2))+1 # We set minimum resolution to be 4
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

    def _make_transes(self, type):
        for i in range(self.num_blocks-1):
            self.transes += self._make_trans(type)

    def _make_trans(self,type):
        return [Scale(type)]

    def precise_layers_per_block(self, i):
        if i == 0: return self.layers_per_block-1 # smallest block has one less layers
        else: return self.layers_per_block

    def _make_block(self, in_ch, out_ch, i, type):
        # smallest block of D has last=1; otherwise, last=0
            return [Block(args=self.args, layers_per_block=self.precise_layers_per_block(i), in_ch=in_ch,
                          out_ch=out_ch, type=type, last=(1-i if type == 'D' else 0))]

    def required_code_length(self): # The length of code to be used
        return M*(self.layers_per_block*self.num_blocks-1)

    def forward(self, x, code):
        return NotImplementedError

class G(Agent):
    def __init__(self, args, resolution, layers_per_block=5):
        super(G, self).__init__(resolution, layers_per_block)
        self.args = args

        # Maybe we should use variable latent dim (dim=512 in ProgGAN)?
        self.pn = PixelNormLayer()

        # stemconv: 1x1 output to 4x4 output
        self.stem_conv = Conv(self.nf(0), self.nf(1), 4, stride=4, padding=0, deconv=True)

        # assert self.num_blocks == 4
        # Note that progGAN used 512(latent),512,512,512,256 for CIFAR
        # The default setting is 512(latent)-> 512 (1st layer)-> 256-> 128-> 64
        self.blocks += self._make_block(self.nf(1), self.nf(1), 0, type='G')
        for i in range(1,self.num_blocks):
            self.blocks += self._make_block(self.nf(i),self.nf(i+1), 1, type='G')

        self._make_transes('G')
        self.end_conv = Conv(self.nf(self.num_blocks), 3, 1, padding=0, linear=True, norm_used=False)


    def forward(self, x, code):
        '''
        :param x: latent noise (batch_size x dimension)
        :param code: Python list consisting of G.required_code_length() elements, each of which is an integer from 1 to 2^R
        :return: x: Batch of generated images (batch_size x 3 x resolution x resolution)
        :return: params: VERY rough estimate of the number of parameters of G
        '''
        code_list = []
        params = 0 # VERY rough estimate of number of parameters
        # assert len(code) = self.required_code_length()
        next_ind = 0
        for i in range(self.num_blocks):
            cur_ind = next_ind
            next_ind = next_ind + M * self.precise_layers_per_block(i)
            code_list.append(code[cur_ind:next_ind])

        x = x.view(x.size(0), -1, 1, 1)
        x = self.pn(x)
        x = self.stem_conv(x)
        params += 4*4*self.nf(0)*self.nf(1)
        for i in range(self.num_blocks):
            x, params_tmp = self.blocks[i](x,code_list[i])
            params += params_tmp
            if i != self.num_blocks-1:
                x = self.transes[i](x)
        x = self.end_conv(x)
        return x, params

class D(Agent):
    def __init__(self, args, resolution, layers_per_block=5):
        super(D, self).__init__(resolution, layers_per_block)
        self.args = args

        self.stem_conv = Conv(3, self.nf(self.num_blocks), 1, padding=0, norm_used=False)

        for i in range(self.num_blocks, 1, -1):
            self.blocks += self._make_block(self.nf(i), self.nf(i-1), 1, type='D')
        self.blocks += self._make_block(self.nf(1), self.nf(1), 0, type='D')  # smallest block has one less layer
        self._make_transes('D')
        self.stddev = MinibatchStatConcatLayer()

        self.end_conv = Conv(self.nf(1), self.nf(0), 4, padding=0, stride=4, norm_used=False)
        self.fc = Linear(self.nf(0), 1)

    def forward(self, x, code):
        '''
        :param x: Batch of generated images (batch_size x 3 x resolution x resolution)
        :param code: Python list consisting of D.required_code_length() elements, each of which is an integer from 1 to 2^R
        :return: x: Scalar output (batch_size x 1)
        :return: params: VERY rough estimate of the number of parameters of D
        '''
        code_list = []
        params = 0 # VERY rough estimate of number of parameters
        # assert len(code) = self.required_code_length()
        next_ind = 0
        for i in range(self.num_blocks):
            cur_ind = next_ind
            next_ind = next_ind + M * self.precise_layers_per_block(self.num_blocks-i-1)
            code_list.append(code[cur_ind:next_ind])
        x = self.stem_conv(x)
        for i in range(self.num_blocks):
            if i == self.num_blocks-1:
                x = self.stddev(x)
            x, params_tmp = self.blocks[i](x,code_list[i])
            params += params_tmp
            if i != self.num_blocks-1:
                x = self.transes[i](x)
        x = self.end_conv(x)
        params += 4*4*self.nf(1)*self.nf(0)
        x = self.fc(x.view(-1,x.size(1)))
        return x, params

class Layer(nn.Module):
    def __init__(self, args, in_ch, out_ch, type):
        super(Layer, self).__init__()
        self.args = args
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.type = type # 'G' or 'D'
        self.conv1 = nn.ModuleList()
        for i in range(R):
            self.conv1 += self._make_layer(kernel=1, padding=0)

        self.conv3 = nn.ModuleList()
        for i in range(R):
            self.conv3 += self._make_layer(kernel=3, padding=1)

        self.avgpool = nn.ModuleList()
        for i in range(R):
            self.avgpool += [nn.AvgPool2d(3, stride=1, padding=1)]

        self.maxpool = nn.ModuleList()
        for i in range(R):
            self.maxpool += [nn.MaxPool2d(3, stride=1, padding=1)]

        if type == 'G':
            self.pn = Dynamic_PixelNormLayer(out_ch)

    def _make_layer(self, kernel, padding):
        return [Conv(self.in_ch, self.out_ch//R, kernel, padding=padding, norm_used=False)]

    # Count the number of active (non-zero) units of channels (each unit of channels consists of out_ch/R channels)
    def count(self, code):
        count = 0
        for i in range(R):
            if code[i] == '1':
                count += 1
        return count

    def forward(self, x, code): # code here is already binarized and partitioned to each branch of layer
        # After passing through each branch, concatenate the channels
        params = 0
        act_ch = 0
        list = []
        for i in range(R):
            if code[0][i] == '1':
                list.append(self.conv1[i](x))
            else:
                if self.args.cuda:    
                    list.append(V(torch.zeros(x.size(0),self.out_ch//R,x.size(2),x.size(3)).cuda(),volatile=not self.training,requires_grad=False))
                else:
                    list.append(V(torch.zeros(x.size(0),self.out_ch//R,x.size(2),x.size(3)),volatile=not self.training,requires_grad=False))
        count = self.count(code[0])
        params += self.in_ch * self.out_ch * count // R
        act_ch += self.out_ch*count//R

        for i in range(R):
            if code[1][i] == '1':
                list.append(self.conv3[i](x))
            else:
                if self.args.cuda: 
                    list.append(V(torch.zeros(x.size(0),self.out_ch//R,x.size(2),x.size(3)).cuda(),volatile=not self.training,requires_grad=False))
                else:
                    list.append(V(torch.zeros(x.size(0),self.out_ch//R,x.size(2),x.size(3)),volatile=not self.training,requires_grad=False))
        count = self.count(code[1])
        params += 3*3*self.in_ch * self.out_ch * count // R
        act_ch += self.out_ch * count // R

        for i in range(R):
            if code[2][i] == '1':
                list.append(F.leaky_relu(self.avgpool[i](x[:,(self.in_ch//R)*i:(self.in_ch//R)*(i+1),:,:]), negative_slope=0.2))
            else:
                if self.args.cuda:
                    list.append(V(torch.zeros(x[:,(self.in_ch//R)*i:(self.in_ch//R)*(i+1),:,:].size()).cuda(),volatile=not self.training,requires_grad=False))
                else:
                    list.append(V(torch.zeros(x[:,(self.in_ch//R)*i:(self.in_ch//R)*(i+1),:,:].size()),volatile=not self.training,requires_grad=False))
        count = self.count(code[2])
        act_ch += self.in_ch * count // R

        for i in range(R):
            if code[3][i] == '1':
                list.append(F.leaky_relu(self.maxpool[i](x[:,(self.in_ch//R)*i:(self.in_ch//R)*(i+1),:,:]), negative_slope=0.2))
            else:
                if self.args.cuda:
                    list.append(V(torch.zeros(x[:,(self.in_ch//R)*i:(self.in_ch//R)*(i+1),:,:].size()).cuda(),volatile=not self.training,requires_grad=False))
                else:
                    list.append(V(torch.zeros(x[:,(self.in_ch//R)*i:(self.in_ch//R)*(i+1),:,:].size()),volatile=not self.training,requires_grad=False))
        count = self.count(code[3])
        act_ch += self.in_ch * count // R

        output = torch.cat(list, 1)

        if self.type == 'G':
            output = self.pn(output, act_ch)

        # For debug
        #print(self.in_ch)
        #print(act_ch)
        #print(output.size())

        return output, params, act_ch

# Note that channel size changes differently for G and D for the sake of symmetry
class Block(nn.Module):
    def __init__(self, args, layers_per_block, in_ch, out_ch, type, last):
        super(Block, self).__init__()
        self.args = args
        self.layers_per_block = layers_per_block
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.type = type # 'D' or 'G'
        self.last = last # stddev changes channel number: last = 1 if the block is last and of D, 0 otherwise

        self.layers = nn.ModuleList()
        for i in range(self.layers_per_block):
            self.layers += self._make_layer(i)

        self.stabilizers = nn.ModuleList()
        for i in range(self.layers_per_block):
            self.stabilizers += self._make_stabilizers(i)

    # To make the structure be similar for G and D, as done in ProgGAN, we have to condition as follows:
    def _make_layer(self, i):
        if self.type == 'G':
            if i == 0:
                return [Layer(self.args, self.in_ch, self.out_ch, type = 'G')]
            else:
                return [Layer(self.args, self.out_ch, self.out_ch, type = 'G')]
        else: #'D'
            if i == self.layers_per_block - 1:
                return [Layer(self.args, self.in_ch, self.out_ch, type = 'D')]
            elif i == 0:
                return [Layer(self.args, self.in_ch+self.last, self.in_ch, type = 'D')]
            else:
                return [Layer(self.args, self.in_ch, self.in_ch, type = 'D')]

    def _make_stabilizers(self, i):
        # i-th stabilizer is located after the i-th layer, not before

        def num_in_ch(i): # The number of in_channels of i-th layer
            if self.type == 'G':
                if i == 0:
                    return self.in_ch
                else:
                    return self.out_ch
            else: #'D'
                if i == 0:
                    return self.in_ch + self.last  # stddev gives an additional channel to in_ch of 0-th layer
                else:
                    return self.in_ch

        def in_ch_stab(i): # The number of in_channels of the i-th stabilizer
            # out_channels of i-th layer
            # the first 2 is the number of possible nxn convs (1x1, 3x3)
            # the next 2 is the number of possible poolings (max, avg)
            if (i == 0 and self.type == 'G') or (i == self.layers_per_block - 1 and self.type == 'D'):
                sum = 2*self.in_ch+2*self.out_ch  # BEWARE that this should be modified if masks to be used are modified
            elif self.type == 'G':
                sum = (M-1)*self.out_ch
            elif self.type == 'D':
                sum = (M-1)*self.in_ch
            else:
                raise ValueError
            # channels from skip connection
            for j in range(i+1):
                sum += num_in_ch(j)
            return sum

        return [Conv(in_ch_stab(i), num_in_ch(i+1) if i != self.layers_per_block-1 else self.out_ch, 1,
                     padding=0, norm_used=(self.type == 'G'))]

    def forward(self, x, code): # code here is not binarized yet

        # Divide the code to the one for skip connections and the one for channel config
        # The code for channel config is translated to binary
        code_list_ch =  [[] for _ in range(self.layers_per_block)]
        code_list_sk = []

        for i in range(self.layers_per_block):
            for j in range(M-1): #1x1 conv, 3x3 conv, maxpool, avgpool
                c = str(bin(code[M * i + j] - 1))[2:]
                c += "0" * (R - len(c))
                code_list_ch[i].append(c)
            str_tmp = str(bin(code[M * i + (M-1)] % (2 ** (self.layers_per_block - i))))[2:]
            str_tmp += "0" * ((self.layers_per_block - i) - len(str_tmp))
            code_list_sk.append(str_tmp)

        # Compose layers and skip connections
        params = 0
        tmp = [] # For saving outputs for skip connections
        for i in range(self.layers_per_block):
            tmp.append(x)
            x, p, act_ch = self.layers[i](x, code_list_ch[i])
            params += p
            inactive_ch = x.size(1)-act_ch
            tmp2 = [x] # For concatenating channels from skip connections
            for j in range(i+1):
                if code_list_sk[j][i-j] == '1':
                    tmp2.append(tmp[j])
                else:
                    if self.args.cuda:
                        tmp2.append(V(torch.zeros(tmp[j].size()).cuda(),volatile=not self.training,requires_grad=False))
                    else:
                        tmp2.append(V(torch.zeros(tmp[j].size()),volatile=not self.training,requires_grad=False))
            x = torch.cat(tmp2, 1)
            del tmp2
            x = self.stabilizers[i](x)
            if self.type == 'G' or (self.type == 'D' and i == self.layers_per_block):
                params += (x.size(1)-inactive_ch)*self.out_ch
            else:
                params += (x.size(1)-inactive_ch)*self.in_ch
        del tmp
        return x, params

class PixelNormLayer(nn.Module):
    def __init__(self):
        super(PixelNormLayer, self).__init__()
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x*x, dim=1, keepdim=True) + 1e-8)

class Dynamic_PixelNormLayer(nn.Module):
    def __init__(self, all_ch):
        super(Dynamic_PixelNormLayer, self).__init__()
        self.all_ch = all_ch

    def forward(self, x, act_ch):
        # act_ch is the number of active (non-zero) channels
        # This modification is necessary, since there are many inactive channels in ENAS
        return x / torch.sqrt((self.all_ch/act_ch)*torch.mean(x*x, dim=1, keepdim=True) + 1e-8)

#-----------------
# Custom linear, conv, convtranspose layers to accommodate weight scaling and pixel norm
# Beware that norm_used=False for D, but Conv sets norm_used=True by default

class Linear(nn.Linear):
    def __init__(self, in_units, out_units):
        super(Linear, self).__init__(in_units, out_units)
        scale = torch.sqrt(torch.mean(self.weight * self.weight))
        self.scale = V(scale.data, requires_grad=False, volatile=not self.training)

    def forward(self, x):
        x = F.linear(x, self.weight / self.scale.view(1,1).expand_as(self.weight), self.bias)
        x = x * self.scale
        return x + torch.unsqueeze(self.bias, 0)

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
            padding, stride=1, deconv=False, linear=False, norm_used=True):
        super(Conv, self).__init__()
        if deconv:
            self.conv = Conv_transpose(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding, linear=linear, norm_used=norm_used)
        else:
            self.conv = Conv_regular(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding, linear=linear, norm_used=norm_used)
    def forward(self, x):
        return self.conv(x)

class Conv_transpose(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size,
            stride=1, padding=0, linear=False, norm_used=True):
        super(Conv_transpose, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        scale = torch.sqrt(torch.mean(self.weight * self.weight))
        self.scale = V(scale.data, requires_grad=False, volatile=not self.training)
        self.bias = None
        if norm_used:
            self.norm = PixelNormLayer()
        self.linear = linear
        self.norm_used = norm_used

    def forward(self, x):
        s = self.weight / self.scale.view(1, 1, 1, 1).expand_as(self.weight)
        x = F.conv_transpose2d(x, self.weight / self.scale.view(1,1,1,1).expand_as(self.weight), stride=self.stride, padding=self.padding)
        x = x * self.scale
        if not self.linear:
            x = F.leaky_relu(x, negative_slope=0.2)
        if self.norm_used:
            x = self.norm(x)
        return x

# Basically the same as Conv_transpose except that it's nn.Conv2d counterpart...
# I'm not sure whether I can merge them, since inheritance would be inconsistent if merged
class Conv_regular(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
            stride=1, padding=0, linear=False, norm_used=True):
        super(Conv_regular, self).__init__(in_channels, out_channels, kernel_size,
            stride=stride, padding=padding)
        scale = torch.sqrt(torch.mean(self.weight * self.weight))
        self.scale = V(scale.data, requires_grad=False, volatile= not self.training)
        self.bias = None
        if norm_used:
            self.norm = PixelNormLayer()
        self.linear = linear
        self.norm_used = norm_used

    def forward(self, x):
        x = F.conv2d(x, self.weight / self.scale.view(1,1,1,1).expand_as(self.weight), stride=self.stride, padding=self.padding)
        x = x * self.scale
        if not self.linear:
            x = F.leaky_relu(x, negative_slope=0.2)
        if self.norm_used:
            x = self.norm(x)
        return x

# This is used for transition only
class Scale(nn.Module):
    def __init__(self, type):
        super(Scale, self).__init__()
        self.type = type
    def forward(self, x):
        if self.type == 'G':
            return F.upsample(x, scale_factor=2)
        else: #'down'
            return F.avg_pool2d(x, kernel_size=2, stride=2, count_include_pad=False)
#--------------------------

class MinibatchStatConcatLayer(nn.Module):
    def __init__(self):
        super(MinibatchStatConcatLayer, self).__init__()

    def square(self,x): return torch.mul(x,x)

    def forward(self, input):
        reps = list(input.size())
        vals = torch.sqrt(torch.mean(self.square(input - torch.mean(input,0,keepdim=True)), 0,keepdim=True) + 1.0e-8)
        vals = torch.mean(vals.view(-1),0,keepdim=True)
        reps[1]=1
        vals = vals.repeat(*reps)
        return torch.cat([input, vals], 1)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal(m.weight.data)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight.data, a=1)
        m.bias.data.zero_()


# For testing:
#g = G(16)
#d = D(16)
#g.apply(weights_init)
#print(g.required_code_length())
#code = []
#for i in range(d.required_code_length()):
#    code.append(255)
#from functools import reduce

#print(g(V(torch.randn(2,512)),code))
#gp = sum([reduce(lambda x, y: x * y, p.size()) for p in g.parameters()])
#print(gp)

#print(g(V(torch.randn(2,512)),code))
