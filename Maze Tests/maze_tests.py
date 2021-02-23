import numpy as np
import random
import torch
#import gym
from os import path
from Agents import PolicyAgent
from Agents import ACAgent
from Agents import ACRecurrentAgent
from Agents import ACHebbRecurrentAgent
#from pole_environment import CartPoleEnv
from pole_environment import CartPoleEnv_Rand_Length
from utils import plotLearning
import configparser
from shutil import copyfile
import argparse
#import pickle
import time
from PIL import Image
import imageio
import os
import platform
import csv
#import matplotlib.pyplot as plt
#import matplotlib.colors as colors
#from matplotlib import cm
#from numpy import genfromtxt


config = configparser.ConfigParser()
config.read('config.ini')
seed =  int(config['DEFAULT']['seed'])
act_fun = [int(x) for x in config.get('NETWORK_ACT', 'act_fun').split(',')]
hsize = [int(x) for x in config.get('NETWORK_SHAPE', 'hsize').split(',')]
##Create the environment
env = CartPoleEnv_Rand_Length(seed)


#The number of inputs. Currently the number of observations plus last reward and last action
isize =env.observation_space.shape[0] + 2;
num_actions = env.action_space.n


#Network characteristic
learning_rate = float(config['NETWORK_PARAMS']['learning_rate'])
learning_rate_policy = float(config['NETWORK_PARAMS']['learning_rate_policy'])
learning_rate_critic = float(config['NETWORK_PARAMS']['learning_rate_critic'])
rec_layer_in = int(config['NETWORK_PARAMS']['rec_layer_in'])
rec_layer_out = int(config['NETWORK_PARAMS']['rec_layer_out'])
batch_size = int(config['NETWORK_PARAMS']['batch_size'])
gamma = float(config['NETWORK_PARAMS']['gamma'])
blossv = float(config['NETWORK_PARAMS']['blossv'])
eps = float(config['NETWORK_PARAMS']['eps'])
weight_decay= float(config['NETWORK_PARAMS']['weight_decay'])
seed = 1
#Network to use
## 0 = Policy Gradient
## 1 = Actor Critic
## 2 = Actor Critic with Reccurent
## 3 = Actor Critic with Hebbian
network = int(config['NETWORK_PARAMS']['network'])

#Other Network characteristics
save_every = int(config['EPISODE']['save_every'])
max_episodes = int(config['EPISODE']['max_episodes'])
length_of_episodes = int(config['EPISODE']['length_of_episodes'])


#The name of the network will be saved and it's location
save_name = "NET_" + str(network)+ "_LR_" + str(learning_rate) + "_LRC_" + str(learning_rate_critic) + "_LRP_" + str(learning_rate_policy) +"_NAME_"+ config['SAVING']['save_name']
data_path = config['SAVING']['data_path']

if not os.path.exists(data_path):
    os.makedirs(data_path)

save_path = data_path+str(save_name)+"/"

if not os.path.exists(save_path):
    os.makedirs(save_path)

copyfile('config.ini',save_path + 'config.ini' )

render  = (config['EPISODE']['render'] == "True")
learn  = (config['NETWORK_PARAMS']['learn'] == "True")
render_eps = int(config['EPISODE']['render_eps'])
target_range_lower = float(config['EPISODE']['target_range_lower'])
target_range_upper = float(config['EPISODE']['target_range_upper'])

#sets the seed
np.random.seed(seed);
random.seed(seed);
torch.manual_seed(seed)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")







def train_maze(paramdict):
    # params = dict(click.get_current_context().params)
    TOTALNBINPUTS = 9+4+4;
    RFSIZE = 3
    ADDITIONALINPUTS = 4
    # TOTALNBINPUTS =  RFSIZE * RFSIZE + ADDITIONALINPUTS + NBNONRESTACTIONS
    print("Starting training...")
    params = {}
    # params.update(defaultParams)
    params.update(paramdict)
    print("Passed params: ", params)
    print(platform.uname())
    # params['nbsteps'] = params['nbshots'] * ((params['prestime'] + params['interpresdelay']) * params['nbclasses']) + params['prestimetest']  # Total number of steps per episode
    suffix = "btchFixmod_" + "".join([str(x) + "_" if pair[0] is not 'nbsteps' and pair[0] is not 'rngseed' and pair[
        0] is not 'save_every' and pair[0] is not 'test_every' and pair[0] is not 'pe' else '' for pair in
                                      sorted(zip(params.keys(), params.values()), key=lambda x: x[0]) for x in pair])[
                             :-1] + "_rngseed_" + str(
        params['rngseed'])  # Turning the parameters into a nice suffix for filenames

    # Initialize random seeds (first two redundant?)
    print("Setting random seeds")
    np.random.seed(params['rngseed']);
    random.seed(params['rngseed']);
    torch.manual_seed(params['rngseed'])

    print("Initializing network")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    net = ACHebbRecurrentAgent([params['lr'], params['lr']], 9+4+4, [params['hs']], 4,  [1,0],0, 0,device, params['gr'], 1e-4, params['l2'], params['blossv'], params['bs'], save_path,"test")

    #print("Shape of all optimized parameters:", [x.size() for x in net.parameters()])
    #allsizes = [torch.numel(x.data.cpu()) for x in net.parameters()]
    #print("Size (numel) of all optimized elements:", allsizes)
    #print("Total size (numel) of all optimized elements:", sum(allsizes))

    # total_loss = 0.0
    print("Initializing optimizer")

    # optimizer = torch.optim.SGD(net.parameters(), lr=1.0*params['lr'])
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=params['gamma'], step_size=params['steplr'])

    BATCHSIZE = params['bs']

    LABSIZE = params['msize']
    lab = np.ones((LABSIZE, LABSIZE))
    CTR = LABSIZE // 2

    # Grid maze
    lab[1:LABSIZE - 1, 1:LABSIZE - 1].fill(0)
    for row in range(1, LABSIZE - 1):
        for col in range(1, LABSIZE - 1):
            if row % 2 == 0 and col % 2 == 0:
                lab[row, col] = 1
    # Not strictly necessary, but cleaner since we start the agent at the
    # center for each episode; may help loclization in some maze sizes
    # (including 13 and 9, but not 11) by introducing a detectable irregularity
    # in the center:
    lab[CTR, CTR] = 0

    all_losses = []
    all_grad_norms = []
    all_losses_objective = []
    all_total_rewards = []
    all_losses_v = []
    lossbetweensaves = 0
    nowtime = time.time()
    meanrewards = np.zeros((LABSIZE, LABSIZE))
    meanrewardstmp = np.zeros((LABSIZE, LABSIZE, params['eplen']))

    pos = 0

    # pw = net.initialZeroPlasticWeights()  # For eligibility traces

    # celoss = torch.nn.CrossEntropyLoss() # For supervised learning - not used here
    with open("Maze_test.csv", "w+", newline = '') as my_csv:
                csvWriter = csv.writer(my_csv, delimiter=',')
                csvWriter.writerow(np.zeros(BATCHSIZE))
    print("Starting episodes!")

    for numiter in range(params['nbiter']):
        net.zero_loss()
        PRINTTRACE = 0
        if (numiter + 1) % (params['pe']) == 0:
            PRINTTRACE = 1

        # lab = makemaze.genmaze(size=LABSIZE, nblines=4)
        # count = np.zeros((LABSIZE, LABSIZE))

        # Select the reward location for this episode - not on a wall!
        # And not on the center either! (though not sure how useful that restriction is...)
        # We always start the episode from the center
        posr = {};
        posc = {}
        rposr = {};
        rposc = {}
        for nb in range(BATCHSIZE):
            # Note: it doesn't matter if the reward is on the center (see below). All we need is not to put it on a wall or pillar (lab=1)
            myrposr = 0;
            myrposc = 0
            while lab[myrposr, myrposc] == 1 or (myrposr == CTR and myrposc == CTR):
                myrposr = np.random.randint(1, LABSIZE - 1)
                myrposc = np.random.randint(1, LABSIZE - 1)
            rposr[nb] = myrposr;
            rposc[nb] = myrposc
            # print("Reward pos:", rposr, rposc)
            # Agent always starts an episode from the center
            posc[nb] = CTR
            posr[nb] = CTR


        reward = np.zeros(BATCHSIZE)
        sumreward = np.zeros(BATCHSIZE)
        rewards = []
        vs = []
        logprobs = []
        dist = 0
        numactionschosen = np.zeros(BATCHSIZE, dtype='int32')

        # reloctime = np.random.randint(params['eplen'] // 4, (3 * params['eplen']) // 4)

        # print("EPISODE ", numiter)
        for numstep in range(params['eplen']):

            inputs = np.zeros((BATCHSIZE, TOTALNBINPUTS), dtype='float32')

            labg = lab.copy()
            for nb in range(BATCHSIZE):
                inputs[nb, 0:RFSIZE * RFSIZE] = labg[posr[nb] - RFSIZE // 2:posr[nb] + RFSIZE // 2 + 1,
                                                posc[nb] - RFSIZE // 2:posc[nb] + RFSIZE // 2 + 1].flatten() * 1.0

                # Previous chosen action
                inputs[nb, RFSIZE * RFSIZE + 1] = 1.0  # Bias neuron
                inputs[nb, RFSIZE * RFSIZE + 2] = numstep / params['eplen']
                inputs[nb, RFSIZE * RFSIZE + 3] = 1.0 * reward[nb]
                inputs[nb, RFSIZE * RFSIZE + ADDITIONALINPUTS + numactionschosen[nb]] = 1

            #inputsC = torch.from_numpy(inputs).to(device)


            actionschosen = net.choose_action(inputs)

            numactionschosen = actionschosen  # We want to break gradients
            reward = np.zeros(BATCHSIZE, dtype='float32')

            for nb in range(BATCHSIZE):
                myreward = 0
                numactionchosen = numactionschosen[nb]

                tgtposc = posc[nb]
                tgtposr = posr[nb]
                if numactionchosen == 0:  # Up
                    tgtposr -= 1
                elif numactionchosen == 1:  # Down
                    tgtposr += 1
                elif numactionchosen == 2:  # Left
                    tgtposc -= 1
                elif numactionchosen == 3:  # Right
                    tgtposc += 1
                else:
                    raise ValueError("Wrong Action")

                reward[nb] = 0.0  # The reward for this step
                if lab[tgtposr][tgtposc] == 1:
                    reward[nb] -= params['wp']
                else:
                    posc[nb] = tgtposc
                    posr[nb] = tgtposr

                # Did we hit the reward location ? Increase reward and teleport!
                # Note that it doesn't matter if we teleport onto the reward, since reward hitting is only evaluated after the (obligatory) move...
                # But we still avoid it.
                if rposr[nb] == posr[nb] and rposc[nb] == posc[nb]:
                    reward[nb] += params['rew']
                    posr[nb] = np.random.randint(1, LABSIZE - 1)
                    posc[nb] = np.random.randint(1, LABSIZE - 1)
                    while lab[posr[nb], posc[nb]] == 1 or (rposr[nb] == posr[nb] and rposc[nb] == posc[nb]):
                        posr[nb] = np.random.randint(1, LABSIZE - 1)
                        posc[nb] = np.random.randint(1, LABSIZE - 1)

            rewards.append(reward)
            net.store_rewards(np.array(reward))
            sumreward += reward

            # This is an "entropy penalty", implemented by the sum-of-squares of the probabilities because our version of PyTorch did not have an entropy() function.
            # The result is the same: to penalize concentration, i.e. encourage diversity in chosen actions.

            # if PRINTTRACE:
            #    print("Step ", numstep, " Inputs (to 1st in batch): ", inputs[0, :TOTALNBINPUTS], " - Outputs(1st in batch): ", y[0].data.cpu().numpy(), " - action chosen(1st in batch): ", numactionschosen[0],
            #            #" - mean abs pw: ", np.mean(np.abs(pw.data.cpu().numpy())),
            #            " -Reward (this step, 1st in batch): ", reward[0])

        # Epi
        with open("Maze_test.csv", "a", newline = '') as my_csv:
                csvWriter = csv.writer(my_csv, delimiter=',')
                csvWriter.writerow(sumreward)
        all_total_rewards.append(sumreward.mean())
        net.learn()
        if (numiter + 1) % params['pe'] == 0:
            print(numiter, "====")
            print("Mean loss: ", lossbetweensaves / params['pe'])
            lossbetweensaves = 0
            print("Mean reward (across batch and last", params['pe'], "eps.): ",
                  np.sum(all_total_rewards[-params['pe']:]) / params['pe'])
            # print("Mean reward (across batch): ", sumreward.mean())
            previoustime = nowtime
            nowtime = time.time()
            print(sumreward)
            print("Time spent on last", params['pe'], "iters: ", nowtime - previoustime)
            
            # print("ETA: ", net.eta.data.cpu().numpy(), " etaet: ", net.etaet.data.cpu().numpy())



def train_maze_1():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rngseed", type=int, help="random seed", default=0)
    parser.add_argument("--rew", type=float,
                        help="reward value (reward increment for taking correct action after correct stimulus)",
                        default=10.0)
    parser.add_argument("--wp", type=float, help="penalty for hitting walls", default=.0)
    parser.add_argument("--bent", type=float,
                        help="coefficient for the entropy reward (really Simpson index concentration measure)",
                        default=0.03)
    parser.add_argument("--blossv", type=float, help="coefficient for value prediction loss", default=.1)
    parser.add_argument("--msize", type=int, help="size of the maze; must be odd", default=5)
    parser.add_argument("--gr", type=float, help="gammaR: discounting factor for rewards", default=.9)
    parser.add_argument("--gc", type=float, help="gradient norm clipping", default=4.0)
    parser.add_argument("--lr", type=float, help="learning rate (Adam optimizer)", default=1e-4)
    parser.add_argument("--eplen", type=int, help="length of episodes", default=200)
    parser.add_argument("--hs", type=int, help="size of the recurrent (hidden) layer", default=100)
    parser.add_argument("--bs", type=int, help="batch size", default=30)
    parser.add_argument("--l2", type=float, help="coefficient of L2 norm (weight decay)", default=0)  # 3e-6
    parser.add_argument("--nbiter", type=int, help="number of learning cycles", default=20000)
    parser.add_argument("--save_every", type=int, help="number of cycles between successive save points", default=50)
    parser.add_argument("--pe", type=int, help="number of cycles between successive printing of information",
                        default=10)
    args = parser.parse_args();
    argvars = vars(args);
    argdict = {k: argvars[k] for k in argvars if argvars[k] != None}

    train_maze(argdict)
#runNetworks(learn, network, save_name, save_every, max_episodes,length_of_episodes, render, render_eps, target_range_lower, target_range_upper)
train_maze_1()
#load_and_plot(0)