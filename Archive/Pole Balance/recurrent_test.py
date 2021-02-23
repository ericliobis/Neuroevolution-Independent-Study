import numpy as np
import random
import torch
import time
import gym
import torch.nn.functional as F
from Regular_Network import RecurrentNetwork
from Regular_Network import RegularNetwork
from utils import plotLearning
import torch.nn as nn

isize = 1 #env.observation_space.shape[0]
seed = 2
hsize = [10,10]
outputs  =2
model_save_name = "pole_balance_1obs_recc"
learning_rate = 0.001
gamma = 0.99


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


reg_test = RegularNetwork(learning_rate, isize, hsize, outputs,device, False)
rec_test = RecurrentNetwork(learning_rate, isize, hsize, outputs, device, False, 1, 1)

criterion = nn.NLLLoss()

score_history_reg = []
score_history_rec = []

num_episodes = 10000
length_episodes = 1000

for ep in range(num_episodes):
    rec_vals = rec_test.initialZeroState()
    start_val = random.random()
    last_val = start_val
    reg_reward = 0
    rec_reward = 0

    reg_perc_right = 0
    rec_perc_right = 0
    loss_rec =torch.tensor([0.]).to(device);

    for i in range(length_episodes):

        diff = random.random() * random.sample([1, -1], 1)[0]
        current_val = last_val + diff
        reg_guess_ = reg_test.forward([[current_val]])
        [rec_guess_, rec_vals] = rec_test.forward([[current_val]], rec_vals)
        reg_guess = F.softmax(reg_guess_)
        rec_guess = F.softmax(rec_guess_)
        #0 is same, 1 is greater, 2 is less

        actual = torch.tensor([0]).to(device);
        n = 0
        if diff<0:
            n = 1
        elif diff >0:
            n = 0

        actual[0] = n
        if (reg_guess[0][0] > 0.5 and n == 0) or (reg_guess[0][1] > 0.5 and n == 1):
            reg_perc_right +=1
        if (rec_guess[0][0] > 0.5 and n == 0) or (rec_guess[0][1] > 0.5 and n == 1):
            rec_perc_right +=1
        loss_reg = criterion(reg_guess, actual)
        loss_rec += criterion(rec_guess, actual)
        reg_reward += loss_reg
        rec_reward += loss_rec
        reg_test.optimizer.zero_grad()



        loss_reg.backward()
        reg_test.optimizer.step()

        last_val = current_val
    rec_test.optimizer.zero_grad()
    loss_rec.backward()
    rec_test.optimizer.step()
    print("Diff:", actual, "Reg:", reg_guess, "Rec:", rec_guess)
    print("Reg Percent:", reg_perc_right/length_episodes, "Rec Percent: ", rec_perc_right/length_episodes)
    #print("Loss reg:",reg_reward,"Loss Rec: ",rec_reward)














