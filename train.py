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

import tflib as lib
import tflib.save_images
import tflib.mnist
import tflib.cifar10
import tflib.plot
import tflib.inception_score

from functools import reduce

import ENAS_GAN
from utils_GAN import generate_image, get_inception_score, preprocess
from storage import RolloutStorage

args = get_args()

use_cuda = args.cuda
DIM = 512 # This overfits substantially; you're probably better off with 64
LAMBDA = 10 # Gradient penalty lambda hyperparameter
iwass_target = 750.0
CRITIC_ITERS = 1 # How many critic iterations per generator iteration
BATCH_SIZE = 16 # Batch size
ITERS = 200000 # How many generator iterations to train for
OUTPUT_DIM = 3072 # Number of pixels in CIFAR10 (3*32*32)

epochs = 300

M1 = 1  # number of paths used to update Shared Parameters omega in Step 1
M2 = 10 # number of paths used to update Policy Parameters theta in Step 2

netG = ENAS_GAN.G(args, 32)
netD = ENAS_GAN.D(args, 32)
#^D should assign high values to real & low values (e.g. 0) to fake

C_DIM = 64
vocab_size = 2**ENAS_GAN.R
cG = rnn_controller.Controller(args=args, dim=C_DIM, vocab_size=vocab_size)
cD = rnn_controller.Controller(args=args, dim=C_DIM, vocab_size=vocab_size)

g_params_total = sum([reduce(lambda x, y: x * y, p.size()) for p in netG.parameters()])
print("g_params_total", g_params_total)

d_params_total = sum([reduce(lambda x, y: x * y, p.size()) for p in netD.parameters()])
print("d_params_total", d_params_total)

c_params_total = sum([reduce(lambda x, y: x * y, p.size()) for p in cG.parameters()])
print("c_params_total", c_params_total, "each")

if use_cuda:
    gpu = 0
if use_cuda:
    netD = netD.cuda(gpu)
    netG = netG.cuda(gpu)
    cG = cG.cuda(gpu)
    cD = cD.cuda(gpu)

one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda(gpu)
    mone = mone.cuda(gpu)

optimizerG = optim.Adam(netG.parameters(), lr=1e-3, betas=(0.0, 0.99))
optimizerD = optim.Adam(netD.parameters(), lr=1e-3, betas=(0.0, 0.99))
optimizerCG = optim.Adam(cG.parameters(), lr=1e-3, eps=1e-5)
optimizerCD = optim.Adam(cD.parameters(), lr=1e-3, eps=1e-5)

'''TODO: This is hardcoded to CIFAR10'''
num_of_data_point = 60000

'''download & unzip "CIFAR-10 python version" from https://www.cs.toronto.edu/~kriz/cifar.html to obtain cifar-10-batches-py/'''
DATA_DIR = 'cifar-10-batches-py/'
train_gen, dev_gen = lib.cifar10.load(BATCH_SIZE, data_dir=DATA_DIR)
def inf_train_gen():
    while True:
        '''TODO: Why is only images (but not targets) returned?'''
        #for images, target in train_gen():
        for images in train_gen():
            # yield images.astype('float32').reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
            yield images
gen = inf_train_gen()

def calc_gradient_penalty(netD, real_data, fake_data, codeD):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, real_data.nelement()//BATCH_SIZE).contiguous().view(BATCH_SIZE, 3, 32, 32)
    alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, codeD)
    disc_interpolates = disc_interpolates[0]

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA / (iwass_target ** 2)
    return gradient_penalty

'''TODO IMMEDIATE: I think L2 is already built into adam opt via w_decay, so is unnecessary.'''
def L2(a, b):
    return 0 if a is None or b is None else torch.mean((a - b)*(a - b))

rolloutsG = RolloutStorage(netG.required_code_length(), M2)
rolloutsD = RolloutStorage(netD.required_code_length(), M2)

'''TODO: make sure all the zero_grad() are in sensible spots'''


save_path = os.path.join(args.save_dir)
try:
    os.makedirs(save_path)
except OSError:
    pass

print()
if args.save:
    print("WARNING: save will overwrite any models with default names in the folder '" + str(args.save_dir))
else:
    print("save is not turned on so models will NOT be saved.")
print()

if args.load:
    print("loading model from checkpoint")
    netG.load_state_dict(torch.load(save_path, "netG" + ".pt"))
    netD.load_state_dict(torch.load(save_path, "netD" + ".pt"))
    cG.load_state_dict(torch.load(save_path, "cG" + ".pt"))
    cD.load_state_dict(torch.load(save_path, "cD" + ".pt"))

for e in range(epochs):
    print("epoch", e)
    print("Step 1")
    '''
    # Step 1: Training the Shared Parameters omega
    '''
    #"""
    for iteration in range(num_of_data_point//(BATCH_SIZE*CRITIC_ITERS)):
        print("iteration", iteration)
        '''TODO: batch sampling of codes'''
        codesG = [[] for _ in range(M1*(CRITIC_ITERS+1))]
        codesD = [[] for _ in range(M1*(CRITIC_ITERS+1))]
        action = V(torch.LongTensor([[0] for _ in range(M1*(CRITIC_ITERS+1))]), volatile=True)
        #action = V(torch.LongTensor([[0] for _ in range(13)]), volatile=True)
        h_state = None
        for i in range(netG.required_code_length()):
            get_value = True if i == netG.required_code_length()-1 else False
            value, action, h_state = cG.act(action, h_state, get_value=get_value)
            for cdx, _c  in enumerate(action.data.squeeze(1).cpu().numpy()):
                codesG[cdx].append(_c)

        action = V(torch.LongTensor([[0] for _ in range(M1*(CRITIC_ITERS+1))]), volatile=True)
        h_state = None
        for i in range(netD.required_code_length()):
            get_value = True if i == netD.required_code_length()-1 else False
            value, action, h_state = cD.act(action, h_state, get_value=get_value)
            for cdx, _c  in enumerate(action.data.squeeze(1).cpu().numpy()):
                codesD[cdx].append(_c)

        start_time = time.time()
        ############################
        # (1) Update D network
        ###########################

        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update & controller update
        for p in netG.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in controller update
        for i in range(CRITIC_ITERS):
            _data = gen.__next__()
            netG.zero_grad()
            netD.zero_grad()
            cG.zero_grad()
            cD.zero_grad()

            # train with real
            _data = _data.reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
            real_data = torch.stack([preprocess(item) for item in _data])

            if use_cuda:
                real_data = real_data.cuda(gpu)
            real_data_v = autograd.Variable(real_data)

            # import torchvision
            # filename = os.path.join("test_train_data", str(iteration) + str(i) + ".jpg")
            # torchvision.utils.save_image(real_data, filename)

            D_real_tmp = netD(real_data_v, codesD[i%(M1*CRITIC_ITERS)])
            D_real_tmp = D_real_tmp[0]
            
            '''L2 is minus because it will be inverted by mone in backwards'''
            D_real = D_real_tmp.mean() - L2(D_real_tmp,0) * args.iwass_epsilon
            D_real.backward(mone)

            # train with fake
            noise = torch.randn(BATCH_SIZE, DIM)
            if use_cuda:
                noise = noise.cuda(gpu)
            noisev = autograd.Variable(noise, volatile=True)  # totally freeze netG
            fake = autograd.Variable(netG(noisev, codesG[i%(M1*CRITIC_ITERS)])[0].data)
            inputv = fake
            D_fake = netD(inputv, codesD[i%(M1*CRITIC_ITERS)])
            D_fake = D_fake[0]
            D_fake = D_fake.mean()
            D_fake.backward(one)

            # train with gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data, codesD[i%(M1*CRITIC_ITERS)])
            gradient_penalty.backward()

            '''TODO: d_reg was originally here, but now it is before gradient penalty because it was causing backward error'''

            # print "gradien_penalty: ", gradient_penalty

            D_cost = D_fake - D_real + gradient_penalty
            Wasserstein_D = D_real - D_fake

            optimizerD.step()
        ############################
        # (2) Update G network
        ###########################
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation
        netG.zero_grad()
        netD.zero_grad()
        cG.zero_grad()
        cD.zero_grad()

        noise = torch.randn(BATCH_SIZE, DIM)
        if use_cuda:
            noise = noise.cuda(gpu)
        noisev = autograd.Variable(noise)
        fake = netG(noisev, codesG[M1*CRITIC_ITERS])
        fake = fake[0]
        G = netD(fake, codesD[M1*CRITIC_ITERS])
        G = G[0]
        G = G.mean()
        G.backward(mone)
        G_cost = -G
        optimizerG.step()

    if args.save:
        save_model = netG
        if args.cuda:
            save_model = copy.deepcopy(netG).cpu()
        torch.save(save_model, os.path.join(save_path, "netG" + ".pt"))

        save_model = netD
        if args.cuda:
            save_model = copy.deepcopy(netD).cpu()
        torch.save(save_model, os.path.join(save_path, "netD" + ".pt"))
        #"""


    print("Step 2")
    '''
    # Step 2: Training the Policy pi(m;theta)
    '''

    '''TODO: make sure all h_state are init correctly and don't carry over from other variables'''
    for iteration in range(num_of_data_point//(BATCH_SIZE*CRITIC_ITERS*M2)):
        print("iteration", iteration)
        codesG = [[] for _ in range(M2*CRITIC_ITERS)]
        action = V(torch.LongTensor([[0] for _ in range(M2*CRITIC_ITERS)]), volatile=True)
        h_state = None
        for i in range(netG.required_code_length()):
            get_value = False
            value, action, h_state = cG.act(action, h_state, get_value=get_value)
            for cdx, _c  in enumerate(action.data.squeeze(1).cpu().numpy()):
                codesG[cdx].append(_c)

        start_time = time.time()
        ############################
        # (1) Update D controller
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # set to False for training controller
        for p in netG.parameters():  # reset requires_grad
            p.requires_grad = False  # set to False for training controller
        for i in range(CRITIC_ITERS):
            optimizerCD.zero_grad()
            D_rewards = []
            action_log_probs_list = []
            dist_entropy_list = []
            codesD = [[] for _ in range(M2)]
            action = V(torch.LongTensor([[0] for _ in range(M2)]))
            h_state = None
            for _i in range(netD.required_code_length()):
                get_value = True if _i == netD.required_code_length()-1 else False
                #value, action, h_state = cD.act(action, h_state, get_value=get_value)
                value, action, h_state, action_log_probs, dist_entropy = cD.act_and_evaluate(V(action.data), h_state, get_value=get_value)
                for cdx, _c  in enumerate(action.data.squeeze(1).cpu().numpy()):
                    codesD[cdx].append(_c)

                action_log_probs_list.append(action_log_probs)
                dist_entropy_list.append(dist_entropy)
            rolloutsD.insert(torch.stack(action_log_probs_list), torch.stack(dist_entropy_list), value)

            for j in range(M2):
                _data = gen.__next__()
                netD.zero_grad()
                #optimizerCD.zero_grad()

                # train with real
                _data = _data.reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
                real_data = torch.stack([preprocess(item) for item in _data])

                if use_cuda:
                    real_data = real_data.cuda(gpu)
                real_data_v = autograd.Variable(real_data)

                # import torchvision
                # filename = os.path.join("test_train_data", str(iteration) + str(i) + ".jpg")
                # torchvision.utils.save_image(real_data, filename)

                D_real_tmp = netD(real_data_v, codesD[j%M2])
                D_real_tmp = D_real_tmp[0]
                D_real = D_real_tmp.mean()
                #D_real.backward(mone)

                # train with fake
                noise = torch.randn(BATCH_SIZE, DIM)
                if use_cuda:
                    noise = noise.cuda(gpu)
                noisev = autograd.Variable(noise, volatile=True)  # totally freeze netG
                fake = autograd.Variable(netG(noisev, codesG[(i*M2+j)%(M2*CRITIC_ITERS)])[0].data)
                #fake = fake[0]
                inputv = fake
                D_fake = netD(inputv, codesD[j%M2])
                D_fake = D_fake[0]
                D_fake = D_fake.mean()
                #D_fake.backward(one)

                # train with gradient penalty
                gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data, codesD[j%M2])

                # print("gradien_penalty: ", gradient_penalty)

                D_reg = L2(D_real_tmp,0) * args.iwass_epsilon  # additional penalty term to keep the scores from drifting too far from zero

                D_cost = D_fake - D_real + gradient_penalty + D_reg
                #D_cost is the negative reward
                D_reward = -D_cost

                #D_rewards.append(D_reward)
                rolloutsD.insert_reward_GAN(j, D_reward.data)

                Wasserstein_D = D_real - D_fake

                '''TODO: discrim's inception loss'''

            '''TODO: discrim's inception loss is at bottom. should it be done here 5 times instead?'''

            cD_loss = -(rolloutsD.logprobs.mean(0).squeeze(1) * V((rolloutsD.rewards_GAN-rolloutsD.avg_reward_GAN)*args.d_GAN_loss_coef)).mean() - rolloutsD.ents.mean(0) * args.entropy_coef

            #just in case gradient penalty steps caused gradients
            optimizerCD.zero_grad()
            
            cD_loss.backward()
            optimizerCD.step()

            rolloutsD.update_avg_reward_GAN()
            '''TODO: is discrim incept reward different than generats incept reward'''
            rolloutsD.update_avg_reward_INCEPT()

        # gets new codesD after controlD update, in order to update controlG 
        codesD = [[] for _ in range(M2)]
        if args.incept_start_epoch >= e:
            action = V(torch.LongTensor([[0] for _ in range(M2)]))
            action_log_probs_list_D_incept = []
            dist_entropy_list_D_incept = []
        else:
            action = V(torch.LongTensor([[0] for _ in range(M2)]), volatile=True)
        h_state_D = None
        for i in range(netD.required_code_length()):
            if args.incept_start_epoch >= e:
                get_value = True if i == netG.required_code_length()-1 else False
                value, action, h_state_D, action_log_probs, dist_entropy = cD.act_and_evaluate(V(action.data), h_state_D, get_value=get_value)
            else:
                get_value = False
                value, action, h_state_D = cD.act(action, h_state_D, get_value=get_value)

            for cdx, _c  in enumerate(action.data.squeeze(1).cpu().numpy()):
                codesD[cdx].append(_c)

            if args.incept_start_epoch >= e:
                action_log_probs_list_D_incept.append(action_log_probs)
                dist_entropy_list_D_incept.append(dist_entropy)
        if args.incept_start_epoch >= e:
            rolloutsD.insert(torch.stack(action_log_probs_list_D_incept), torch.stack(dist_entropy_list_D_incept), value)

        codesG = [[] for _ in range(M2)]
        action = V(torch.LongTensor([[0] for _ in range(M2)]))
        action_log_probs_list = []
        dist_entropy_list = []
        h_state_G = None
        for i in range(netG.required_code_length()):
            get_value = True if i == netG.required_code_length()-1 else False
            value, action, h_state_G, action_log_probs, dist_entropy = cG.act_and_evaluate(V(action.data), h_state_G, get_value=get_value)
            for cdx, _c  in enumerate(action.data.squeeze(1).cpu().numpy()):
                codesG[cdx].append(_c)

            action_log_probs_list.append(action_log_probs)
            dist_entropy_list.append(dist_entropy)
        rolloutsG.insert(torch.stack(action_log_probs_list), torch.stack(dist_entropy_list), value)

        ############################
        # (2) Update G controller
        ###########################
        '''TODO: should these be the controller params or the GAN params'''
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation
        for p in netG.parameters():  # reset requires_grad
            p.requires_grad = True  # set to False for training controller

        optimizerCG.zero_grad()
        G_rewards = []
        for j in range(M2):
            netG.zero_grad()
            noise = torch.randn(BATCH_SIZE, DIM)
            if use_cuda:
                noise = noise.cuda(gpu)
            noisev = autograd.Variable(noise)
            fake = netG(noisev, codesG[j%M2])
            fake = fake[0]
            G = netD(fake, codesD[j%M2])
            G = G[0]
            G = G.mean()
            #G.backward(mone)
            G_cost = -G
            #G_cost is the negative reward
            G_reward = -G_cost

            #G_rewards.append(G_reward)
            #print('G_reward.data', G_reward.data)
            rolloutsG.insert_reward_GAN(j, G_reward.data)

            #inception portion
            if args.incept_start_epoch >= e:
                '''TODO: will this relu throw things off or will optimization eventually self-correct?'''
                fake = F.relu(fake)
                incept_inp = fake.cpu().data.numpy()
                incept_inp = np.multiply(np.add(np.multiply(incept_inp, 0.5), 0.5), 255).astype('int32')
                incept_inp = incept_inp.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
                #print("incept_inp", list(incept_inp))
                G_incept_reward_mean, G_incept_reward_std = lib.inception_score.get_inception_score(list(incept_inp))
                '''^bigger incept score is better, so reward is positive.
                if it was FID instead, then reward would negative of it because smaller FID is better
                '''

                '''TODO: Should G_incept_reward_std be used for anything?'''
                #print('G_incept_reward_mean', float(G_incept_reward_mean))

                #print('G_incept_reward', G_incept_reward)
                rolloutsG.insert_reward_INCEPT(j, torch.Tensor([float(G_incept_reward_mean)]))

        '''TODO IMMEDIATE: MAKE SURE ROLLOUT STORAGE COPY DOESN'T DETACH GRADIENTS/GRAPH'''

        cG_loss = -(rolloutsG.logprobs.mean(0).squeeze(1) * V((rolloutsG.rewards_GAN-rolloutsG.avg_reward_GAN)*args.g_GAN_loss_coef+(rolloutsG.rewards_INCEPT-rolloutsG.avg_reward_INCEPT)*args.g_Incept_loss_coef)).mean() - rolloutsG.ents.mean(0) * args.entropy_coef

        '''^TODO IMMEDIATE: MAKE SURE mean &/or sum of logprobs & ents are correct'''

        cG_loss.backward()
        optimizerCG.step()

        rolloutsG.update_avg_reward_GAN()
        #rolloutsG.update_avg_reward_INCEPT()

        #update Dcontroller with inception loss diff after Gupdate
        if args.incept_start_epoch >= e:
            for p in netD.parameters():
                p.requires_grad = True  # to avoid computation
            prev_reward_G = rolloutsG.rewards_INCEPT

            optimizerCG.zero_grad()
            optimizerCD.zero_grad()
            codesG = [[] for _ in range(M2)]
            action = V(torch.LongTensor([[0] for _ in range(M2)]))
            action_log_probs_list = []
            dist_entropy_list = []
            h_state_G = None
            for i in range(netG.required_code_length()):
                get_value = True if i == netG.required_code_length()-1 else False
                value, action, h_state_G = cG.act(action, h_state_G, get_value=get_value)
                for cdx, _c  in enumerate(action.data.squeeze(1).cpu().numpy()):
                    codesG[cdx].append(_c)

            for j in range(M2):
                netG.zero_grad()
                noise = torch.randn(BATCH_SIZE, DIM)
                if use_cuda:
                    noise = noise.cuda(gpu)
                noisev = autograd.Variable(noise)
                fake = netG(noisev, codesG[j%M2])
                fake = fake[0]

                fake = F.relu(fake)
                incept_inp = fake.cpu().data.numpy()
                incept_inp = np.multiply(np.add(np.multiply(incept_inp, 0.5), 0.5), 255).astype('int32')
                incept_inp = incept_inp.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
                D_incept_reward_mean, D_incept_reward_std = lib.inception_score.get_inception_score(list(incept_inp))
                '''^bigger incept score is better, so reward is positive.
                if it was FID instead, then reward would negative of it because smaller FID is better
                '''

                '''TODO: Should D_incept_reward_std be used for anything?'''

                rolloutsD.insert_reward_INCEPT(j, torch.Tensor([float(D_incept_reward_mean)]))

            '''TODO IMMEDIATE: should you also use avg baseline on diff of INCEPT loss between updates'''
            #cD_loss = -(rolloutsD.logprobs.mean(0).squeeze(1) * ((rolloutsD.rewards_INCEPT-rolloutsG.rewards_INCEPT)) * args.d_Incept_loss_coef).mean() - rolloutsD.ents.mean(0) * args.entropy_coef
            cD_loss = -(rolloutsD.logprobs.mean(0).squeeze(1) * V((rolloutsD.rewards_INCEPT-rolloutsG.rewards_INCEPT) - (rolloutsD.avg_reward_INCEPT-rolloutsG.avg_reward_INCEPT)) * args.d_Incept_loss_coef).mean() - rolloutsD.ents.mean(0) * args.entropy_coef

            '''^TODO IMMEDIATE: MAKE SURE mean &/or sum of logprobs & ents are correct'''

            #just in case gradient penalty steps caused gradients
            optimizerCD.zero_grad()

            cD_loss.backward()
            optimizerCD.step()

            #rolloutsD.update_avg_reward_GAN()
            rolloutsD.update_avg_reward_INCEPT()

            rolloutsG.update_avg_reward_INCEPT()

    if args.save:
        save_model = cG
        if args.cuda:
            save_model = copy.deepcopy(cG).cpu()
        torch.save(save_model, os.path.join(save_path, "cG" + ".pt"))

        save_model = cD
        if args.cuda:
            save_model = copy.deepcopy(cD).cpu()
        torch.save(save_model, os.path.join(save_path, "cD" + ".pt"))
