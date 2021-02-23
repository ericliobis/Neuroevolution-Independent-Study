from Regular_Network import RegularNetwork
from Regular_Network import ActFunction
from Regular_Network import RecurrentNetwork
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

class Agent(object):
    def __init__(self,cuda, lr,input_dims,gamma = 0.99, n_actions =2, hsize = [10,2,2], model_save_name = "model"  ):
        self.gamma = gamma
        self.model_save_name = model_save_name
        self.reward_memory = []
        self.action_memory = []
        self.policy = RegularNetwork(lr[0], input_dims, hsize, n_actions, cuda, True)

    def choose_action(self, observation):
        probabilties = self.policy.forward(observation)
        action_probs = torch.distributions.Categorical(probabilties)
        action = action_probs.sample();
        self.log_probs = action_probs.log_prob(action)
        self.action_memory.append(self.log_probs)

        return action.item()

    def store_rewards(self,reward):
        self.reward_memory.append(reward)

class PolicyAgent(Agent):


    def learn(self):
        self.policy.optimizer.zero_grad()
        G = np.zeros_like(self.reward_memory, dtype = np.float64)
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum +=self.reward_memory[k]*discount
                discount *= self.gamma
            G[t] = G_sum
        mean = np.mean(G)
        std = np.std(G) if np.std(G)>0 else 1
        G = (G-mean)/std

        G = torch.tensor(G, dtype = torch.float).to(self.policy.device)

        loss = 0

        for g, logprob in zip(G, self.action_memory):
            loss += -g*logprob

        loss.backward()

        self.policy.optimizer.step()

        self.action_memory = []
        self.reward_memory = []
    def save_model(self):
        torch.save(self.policy.state_dict(), "policyagent_"+self.model_save_name)

class ACAgent(Agent):
    def __init__(self, cuda,lr, input_dims, gamma = 0.99,n_actions =2, hsize = [10,2,2] , model_save_name = "model"):

        super(Agent, self).__init__()
        self.gamma = gamma
        self.log_probs = None
        self.model_save_name = model_save_name


        self.policy = RegularNetwork(lr[0], input_dims, hsize, n_actions, cuda, True)

        self.critic = RegularNetwork(lr[1], input_dims, hsize, 1, cuda, False)
        print(self.policy)
        print(self.critic)

        self.reward_memory = []
        self.action_memory = []
    #choose action is the same

    def learn(self, state, reward, new_state,done):
        self.policy.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        critic_value = self.critic.forward(state)
        critic_value_ = self.critic.forward(new_state)
        delta = ((reward + self.gamma*critic_value_*(1-int(done)))-critic_value)

        actor_loss = -self.log_probs * delta
        critic_loss = delta ** 2


        (actor_loss + critic_loss).backward()

        self.policy.optimizer.step()
        self.critic.optimizer.step()
    def save_model(self):
        torch.save(self.policy.state_dict(), "policy_"+self.model_save_name)
        torch.save(self.critic.state_dict(), "critic_"+self.model_save_name)

class ACRecurrentAgent(Agent):
    def __init__(self, cuda,lr, input_dims, gamma = 0.99,n_actions =2, hsize = [10,2,2], rec_layer_in = 1 , rec_layer_out = 2, model_save_name = "model_recc"):
        super(Agent, self).__init__()
        self.gamma = gamma
        self.cuda = cuda
        self.log_probs = None
        self.model_save_name = model_save_name

        self.policy = RecurrentNetwork(lr[0], input_dims, hsize, n_actions, cuda, True, rec_layer_in, rec_layer_out)
        self.policy_rec = self.policy.initialZeroState()
        self.critic = RecurrentNetwork(lr[1], input_dims, hsize, 1, cuda, False,rec_layer_in, rec_layer_out)
        self.critic_rec =  self.critic.initialZeroState()
        self.critic_rec_ = self.critic.initialZeroState()
        print(self.policy)
        print(self.critic)

        self.reward_memory = []
        self.action_memory = []
    #choose action is the same

    def zero_loss(self):
        self.actor_loss = torch.tensor([0.]).to(self.cuda);
        self.critic_loss =  torch.tensor([0.]).to(self.cuda);

    def add_loss(self, state, reward, new_state,done):
        [critic_value, self.critic_rec] = self.critic.forward(state, self.critic_rec)
        [critic_value_, self.critic_rec_] = self.critic.forward(new_state, self.critic_rec)
        delta = ((reward + self.gamma * critic_value_ * (1 - int(done))) - critic_value)
        self.actor_loss += -self.log_probs * delta
        self.critic_loss += delta ** 2

    def choose_action(self, observation):
        [probabilties,self.policy_rec] = self.policy.forward(observation, self.policy_rec)
        action_probs = torch.distributions.Categorical(probabilties)
        action = action_probs.sample();
        self.log_probs = action_probs.log_prob(action)
        self.action_memory.append(self.log_probs)
        return action.item()

    def learn(self):
        self.policy.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()




        (self.actor_loss + self.critic_loss).backward()  # )

        self.policy.optimizer.step()
        self.critic.optimizer.step()
    def save_model(self):
        torch.save(self.policy.state_dict(), "policy_"+self.model_save_name)
        torch.save(self.critic.state_dict(), "critic_"+self.model_save_name)

    def clear(self):
        #self.log_probs.detach_()
        self.policy_rec = self.policy.initialZeroState()


