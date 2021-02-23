import gym
from pole_environment import CartPoleEnv
from Regular_Network import RegularNetwork
from Regular_Network import ActFunction
from utils import discount_and_normalize_rewards
import argparse
import pdb
#from line_profiler import LineProfiler
import torch
from torchvision import models
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from numpy import random
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
import random
import sys
import pickle
import time
import os
import platform

import numpy as np



env = CartPoleEnv()


# Variable Definitions
isize = env.observation_space.shape[0]
seed = 1
hsize = [10,2,2]
num_actions = env.action_space.n
max_episodes = 1
learning_rate = 0.01
gamma = 0.95

print(len(hsize))


#sets the seed
np.random.seed(seed);
random.seed(seed);
torch.manual_seed(seed)

print("Initializing network")
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

net = RegularNetwork(isize, hsize, num_actions, ActFunction.RELU,device).to(device)  # Creating the network
print(net)

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


allRewards = []
total_rewards = 0
maximumRewardRecorded = 0
episode_states, episode_actions, episode_rewards = [],[],[]
for i_episode in range(max_episodes):
    episode_rewards_sum = 0
    state = env.reset()



    for t in range(00):
        env.render()

        ##First run through of the network. Take the observation, put it through the network
        action_dist = net(state).cpu().detach().numpy()

        action = np.random.choice(action_dist.shape[0], p = action_dist)

        new_state, reward, done, info = env.step(action)


        ######RECORDS AND SAVES THE STATE#####
        episode_states.append(state)

        #which action was taken
        action_ = np.zeros(num_actions)
        action_[action] = 1
        episode_actions.append(action_)

        episode_rewards.append(reward)
        #######################################


        if done:
            episode_rewards_sum = np.sum(episode_rewards)

            allRewards.append(episode_rewards_sum)

            total_rewards = np.sum(allRewards)

            mean_reward = np.divide(total_rewards, i_episode+1)

            maximumRewardRecorded = np.amax(allRewards)

            print("==========================================")
            print("Episode: ", i_episode)
            print("Reward: ", episode_rewards_sum)
            print("Mean Reward", mean_reward)
            print("Max reward so far: ", maximumRewardRecorded)

            discounted_episode_rewards = discount_and_normalize_rewards(episode_rewards,gamma)
            print(discounted_episode_rewards)

            #loss =
            break
env.close()




