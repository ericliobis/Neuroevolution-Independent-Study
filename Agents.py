from Networks import *
import argparse
import pdb
#from line_profiler import LineProfiler
import torch
#from torchvision import models
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
import csv

####################################
#Regular Feed Forward Network Agent#
####################################
class Regular_Agent(object):
    def __init__(self,lr,isize,hsizes, num_actions,act_functions, cuda, gamma = 0.99, eps = 1e-4, weight_decay = 0 , batch_size= 1, save_path = "",model_save_name = "FFmodel"  ):
        # lr (float) = learning rate
        # isize (int) = input size
        # hsizes (int []) = sizes of the hidden layers
        # num_actions (int) = output size (number of actions)
        # act_functions (int []) - activation function for each layer. 0 in None (generally used for last layer), 1 is TANH, 2 is SIGMOID, 3 is RELU
        # cuda {"cuda" or "cpu")
        # gamma (float) - discounting factor for rewards
        # eps (float) - epsilon value for Adam optimizer - DEFAULT: 1e-4
        # weight decay (float) - weight decay for Adam optimizer - DEFAULT: 0
        # batch_size (int) - the batch size the network - DEFAULT: 1
        # save_path (string) - the path to save to  - DEFAULT: ""
        # model_save_name (string) - the name of the model - DEFAULT: FFmodel
        self.gamma = gamma
        self.cuda = cuda
        self.bs= batch_size;
        self.model_save_name = model_save_name
        self.reward_memory = []
        self.action_memory = []
        self.policy = RegularNetwork(lr[0], isize, hsizes, num_actions, act_functions, cuda, eps, weight_decay,batch_size)
        self.save_path = save_path
        print(self.policy)

    def choose_action(self, observation):
        probabilties = self.policy.forward(torch.from_numpy(observation).to(self.cuda))
        probabilties = torch.softmax(probabilties, dim=1)
        action_probs = torch.distributions.Categorical(probabilties)
        action = action_probs.sample();
        self.log_probs = action_probs.log_prob(action)
        self.action_memory.append(self.log_probs)

        return action.cpu().numpy()

    def store_rewards(self,reward):
        self.reward_memory.append(reward)

    def save_model(self):
        torch.save(self.policy.state_dict(), self.save_path+ "Regular_agent_" + self.model_save_name)
    def load_model(self, model_path):
        self.policy.load_state_dict(torch.load(model_path))

###############################
#Policy Gradient Network Agent#
###############################

class PolicyAgent(Regular_Agent):
    def learn(self):
        #Policy Gradient algorithm
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
    def zero_loss(self):
        #nothing there
        x=0;


####################
#Actor Critic Agent#
####################

class ACAgent(object):

    def __init__(self,lr,isize,hsizes, num_actions,act_functions, cuda, gamma = 0.99, eps = 1e-4, weight_decay = 0 ,blossv = 0.1, batch_size= 1,  save_path = "",model_save_name = "ACmodel"  ):
        # lr (float []) = learning rate [0]- policy, [1] - critic
        # isize (int) = input size
        # hsizes (int []) = sizes of the hidden layers
        # num_actions (int) = output size (number of actions)
        # act_functions (int []) - activation function for each layer. 0 in None (generally used for last layer), 1 is TANH, 2 is SIGMOID, 3 is RELU
        # cuda {"cuda" or "cpu")
        # gamma (float) - discounting factor for rewards
        # eps (float) - epsilon value for Adam optimizer - DEFAULT: 1e-4
        # weight decay (float) - weight decay for Adam optimizer - DEFAULT: 0
        # batch_size (int) - the batch size the network - DEFAULT: 1
        # save_path (string) - the path to save to  - DEFAULT: ""
        # model_save_name (string) - the name of the model - DEFAULT: ACmodel
        self.gamma = gamma
        self.cuda = cuda
        self.blossv = blossv
        self.bs = batch_size
        self.model_save_name = model_save_name
        self.rewards = []

        self.policy = RegularNetwork(lr[0], isize, hsizes, num_actions, act_functions, cuda, eps, weight_decay)
        self.critic = RegularNetwork(lr[1], isize, hsizes, 1, act_functions, cuda, eps, weight_decay)
        self.zero_loss()
        self.save_path = save_path
        print(self.policy)
        print(self.critic)


    def choose_action(self, observation):
        probabilties = self.policy.forward(torch.from_numpy(observation).float().to(self.cuda))
        v = self.critic.forward(torch.from_numpy(observation).float().to(self.cuda))

        probabilties = torch.softmax(probabilties, dim=1)
        self.vs.append(v)
        action_probs = torch.distributions.Categorical(probabilties)
        action = action_probs.sample();
        self.log_probs.append(action_probs.log_prob(action))
        return action.cpu().numpy()

    def store_rewards(self,reward):
        self.rewards.append(reward)
    # def learn(self, state, reward, new_state,done):
    #
    #
    #     critic_value = self.critic.forward(state)
    #     critic_value_ = self.critic.forward(new_state)
    #     delta = ((reward + self.gamma*critic_value_*(1-int(done)))-critic_value)
    #
    #     actor_loss = -self.log_probs * delta
    #     critic_loss = delta ** 2
    #
    #
    #     (actor_loss + critic_loss).backward()
    #
    #     self.policy.optimizer.step()
    #     self.critic.optimizer.step()
    def learn(self):
        eplen = len(self.rewards)
        self.policy.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        self.R = torch.zeros(self.bs).to(self.cuda)

        for numstepb in reversed(range(eplen)):
            self.R = self.gamma * self.R + torch.from_numpy(self.rewards[numstepb]).to(self.cuda)

            ctrR = self.R - self.vs[numstepb][0]
            self.lossv += ctrR.pow(2).sum() / self.bs
            self.loss -= (self.log_probs[numstepb] * ctrR.detach()).sum() / self.bs

        self.loss += self.blossv * self.lossv
        self.loss /= eplen
        self.loss.backward()
        self.critic.optimizer.step()
        self.policy.optimizer.step()

    def zero_loss(self):
        self.R = torch.zeros(self.bs).to(self.cuda)
        self.rewards = []
        self.vs = []
        self.loss = 0
        self.lossv = 0
        self.log_probs = []
        self.critic.optimizer.zero_grad()
        self.policy.optimizer.zero_grad()


    def save_model(self):
        torch.save(self.policy.state_dict(), self.save_path+"policy_"+self.model_save_name)
        torch.save(self.critic.state_dict(), self.save_path+"critic_"+self.model_save_name)

    def load_model(self, model_path_policy, model_path_critic):
        self.policy.load_state_dict(torch.load(model_path_policy))
        self.critic.load_state_dict(torch.load(model_path_critic))






class ACRecurrentAgent(object):
    def __init__(self, lr, isize, hsizes, num_actions,act_functions, rec_layer_out, rec_layer_in, cuda, gamma = 0.99,  eps = 1e-4, weight_decay = 0, blossv = 0.1, batch_size = 1, save_path = "",model_save_name = "ACmodel_rec"  ):
        # lr (float []) = learning rate [0]- policy, [1] - critic
        # isize (int) = input size
        # hsizes (int []) = sizes of the hidden layers
        # num_actions (int) = output size (number of actions)
        # act_functions (int []) - activation function for each layer. 0 in None (generally used for last layer), 1 is TANH, 2 is SIGMOID, 3 is RELU
        # rec_layer_out (int) - The layer where the recurrent values go out (I.E. 0 is after the first layer activation function)
        # rec_layer_in (int) - The layer where the recurrent values come in (I.E. 0 is before the first layer activation function)
        # cuda {"cuda" or "cpu")
        # gamma (float) - discounting factor for rewards
        # eps (float) - epsilon value for Adam optimizer - DEFAULT: 1e-4
        # weight decay (float) - weight decay for Adam optimizer - DEFAULT: 0
        # blossv (float) - coefficient for value prediction loss - DEFAULT: 0.1
        # batch_size (int) - the batch size the network - DEFAULT: 1
        # save_path (string) - the path to save to  - DEFAULT: ""
        # model_save_name (string) - the name of the model - DEFAULT: ACmodel_rec
        self.gamma = gamma
        self.cuda = cuda
        self.model_save_name = model_save_name
        self.blossv = blossv
        self.reward_memory = []
        self.action_memory = []
        self.bs = batch_size

        self.policy = RecurrentNetwork(lr[0], isize, hsizes, num_actions, act_functions, rec_layer_out, rec_layer_in, cuda, eps, weight_decay, batch_size)
        self.policy_rec = self.policy.initialZeroState()
        self.critic = RecurrentNetwork(lr[1], isize, hsizes, 1, act_functions, rec_layer_out, rec_layer_in, cuda, eps, weight_decay, batch_size)
        self.critic_rec =  self.critic.initialZeroState()
        self.critic_rec_ = self.critic.initialZeroState()
        self.save_path = save_path
        print(self.policy)
        print(self.critic)
        self.zero_loss()


    def zero_loss(self):
        #self.actor_loss = torch.tensor([0.]).to(self.cuda);
        #self.critic_loss = torch.tensor([0.]).to(self.cuda);
        self.R = torch.zeros(self.bs).to(self.cuda)
        self.rewards = []
        self.vs = []
        self.loss = 0
        self.lossv = 0
        self.log_probs = []
        self.critic.optimizer.zero_grad()
        self.policy.optimizer.zero_grad()

        self.critic_rec = self.critic.initialZeroState().to(self.cuda)
        self.policy_rec = self.policy.initialZeroState().to(self.cuda)

    def store_rewards(self, reward):
        self.rewards.append(reward.copy())

    def choose_action(self, observation):
        [probabilties,self.policy_rec] = self.policy.forward(torch.from_numpy(observation).to(self.cuda), self.policy_rec)

        v, self.critic_rec = self.critic(torch.from_numpy(observation).to(self.cuda), self.critic_rec)
        probabilties = torch.softmax(probabilties, dim=1)
        self.vs.append(v)
        action_probs = torch.distributions.Categorical(probabilties)
        action = action_probs.sample();

        self.log_probs.append(action_probs.log_prob(action))
        return action.cpu().numpy()

    def learn(self):
        eplen = len(self.rewards)
        self.R = torch.zeros(self.bs).to(self.cuda)

        for numstepb in reversed(range(eplen)):
            self.R = self.gamma * self.R + torch.from_numpy(self.rewards[numstepb]).to(self.cuda)

            ctrR = self.R - self.vs[numstepb][0]
            self.lossv += ctrR.pow(2).sum() / self.bs
            self.loss -= (self.log_probs[numstepb] * ctrR.detach()).sum() / self.bs

        self.loss += self.blossv * self.lossv
        self.loss /= eplen
        self.loss.backward()
        self.critic.optimizer.step()
        self.policy.optimizer.step()
    def save_model(self):
        torch.save(self.policy.state_dict(), self.save_path+"policy_"+self.model_save_name)
        torch.save(self.critic.state_dict(), self.save_path+"critic_"+self.model_save_name)
    def load_model(self, model_path_policy, model_path_critic):
        self.policy.load_state_dict(torch.load(model_path_policy))
        self.critic.load_state_dict(torch.load(model_path_critic))



class ACHebbRecurrentAgent(object):
    def __init__(self, lr, isize, hsizes, num_actions,act_functions, rec_layer_out, rec_layer_in, cuda, gamma = 0.99,  eps = 1e-4, weight_decay = 0, blossv = 0.1, batch_size = 1, save_path = "",model_save_name = "ACmodel_hebb"  ):
        # lr (float []) = learning rate [0]- policy, [1] - critic
        # isize (int) = input size
        # hsizes (int []) = sizes of the hidden layers
        # num_actions (int) = output size (number of actions)
        # act_functions (int []) - activation function for each layer. 0 in None (generally used for last layer), 1 is TANH, 2 is SIGMOID, 3 is RELU
        # rec_layer_out (int) - The layer where the recurrent values go out (I.E. 0 is after the first layer activation function)
        # rec_layer_in (int) - The layer where the recurrent values come in (I.E. 0 is before the first layer activation function)
        # cuda {"cuda" or "cpu")
        # gamma (float) - discounting factor for rewards
        # eps (float) - epsilon value for Adam optimizer - DEFAULT: 1e-4
        # weight decay (float) - weight decay for Adam optimizer - DEFAULT: 0
        # blossv (float) - coefficient for value prediction loss - DEFAULT: 0.1
        # batch_size (int) - the batch size the network - DEFAULT: 1
        # save_path (string) - the path to save to  - DEFAULT: ""
        # model_save_name (string) - the name of the model - DEFAULT: ACmodel_hebb
        self.gamma = gamma
        self.cuda = cuda
        self.bs = batch_size
        self.blossv = blossv
        self.model_save_name = model_save_name
        self.save_path = save_path
        self.writeWeights = False;

        self.policy = HebbianNetwork(lr[0], isize, hsizes, num_actions,act_functions,rec_layer_out, rec_layer_in, cuda,eps , weight_decay , batch_size)
        self.critic = HebbianNetwork(lr[1], isize, hsizes, 1,act_functions,rec_layer_out, rec_layer_in, cuda,eps, weight_decay, batch_size)
        if(self.writeWeights):
            with open("policy_params.csv", "w+", newline = '') as my_csv:
                csvWriter = csv.writer(my_csv, delimiter=',')
                dummy = []
                for param in self.policy.parameters():
                    dummy1 = param.data.cpu().detach().numpy()
                    for p1 in dummy1:
                        if(np.isscalar(p1)):
                    
                            dummy.append(p1)
                        else:
                            for p2 in p1:
                        
                                dummy.append(p2)
                csvWriter.writerow(dummy)
            with open("critic_params.csv", "w+", newline = '') as my_csv:
                csvWriter = csv.writer(my_csv, delimiter=',')
                dummy = []
                for param in self.critic.parameters():
                    dummy1 = param.data.cpu().detach().numpy()
                    for p1 in dummy1:
                        if(np.isscalar(p1)):
                        
                            dummy.append(p1)
                        else:
                            for p2 in p1:
                            
                                dummy.append(p2)
                csvWriter.writerow(dummy)
        print(self.policy)
        print(self.critic)
        self.zero_loss()

    def zero_loss(self):
        self.actor_loss = torch.tensor([0.]).to(self.cuda);
        self.critic_loss =  torch.tensor([0.]).to(self.cuda);
        self.R = torch.zeros(self.bs).to(self.cuda)
        self.rewards = []
        self.vs = []
        self.loss = 0
        self.lossv = 0
        self.log_probs = []
        self.critic.optimizer.zero_grad()
        self.policy.optimizer.zero_grad()

        self.critic_rec =  self.critic.initialZeroState().to(self.cuda)
        self.critic_rec_hebb = self.critic.initialZeroHebb().to(self.cuda)
        self.policy_rec = self.policy.initialZeroState().to(self.cuda)
        self.policy_rec_hebb = self.policy.initialZeroHebb().to(self.cuda)


    def store_rewards(self, reward):

        #Get the critics thought at the last state
        #[critic_value, self.critic_rec, self.critic_rec_hebb] = self.critic.forward(state, self.critic_rec, self.critic_rec_hebb)
        #self.vs.append(critic_value)
        self.rewards.append(reward.copy())

    def choose_action(self, observation):
        #[probabilties,self.policy_rec, self.policy_rec_hebb, v] = self.policy(observation, self.policy_rec,self.policy_rec_hebb)
        #print("policy rec:", self.policy_rec)
        #print("policy rec hebb:", self.policy_rec_hebb)

        probabilties, self.policy_rec, self.policy_rec_hebb = self.policy(torch.from_numpy(observation).to(self.cuda), self.policy_rec, self.policy_rec_hebb)

        v, self.critic_rec, self.critic_rec_hebb = self.critic(torch.from_numpy(observation).to(self.cuda), self.critic_rec, self.critic_rec_hebb)

        #print("before",probabilties)
        probabilties = torch.softmax(probabilties, dim=1)
        #print("after", probabilties)
        self.loss += (0.03 * probabilties.pow(2).sum() / self.bs)
        self.vs.append(v)
        action_probs = torch.distributions.Categorical(probabilties)

        action = action_probs.sample();
        self.log_probs.append(action_probs.log_prob(action))
        return action.cpu().numpy()

    def learn(self):
        eplen = len(self.rewards)
        self.R = torch.zeros(self.bs).to(self.cuda)

        for numstepb in reversed(range(eplen)):
            self.R = self.gamma * self.R + torch.from_numpy(self.rewards[numstepb]).to(self.cuda)

            ctrR = self.R - self.vs[numstepb][0]
            self.lossv += ctrR.pow(2).sum() / self.bs
            self.loss -= (self.log_probs[numstepb] * ctrR.detach()).sum() / self.bs

        self.loss += self.blossv * self.lossv
        self.loss /= eplen
        self.loss.backward()
        self.critic.optimizer.step()
        self.policy.optimizer.step()
        self.save_params()
        return self.loss
    def save_model(self):
        torch.save(self.policy.state_dict(), self.save_path+"policy_"+self.model_save_name)
        torch.save(self.critic.state_dict(), self.save_path+"critic_"+self.model_save_name)
    def load_model(self, model_path_policy, model_path_critic):
        self.policy.load_state_dict(torch.load(model_path_policy))
        self.critic.load_state_dict(torch.load(model_path_critic))
    def save_params(self):
        if(self.writeWeights):
            with open("policy_params.csv", "a", newline = '') as my_csv:
                csvWriter = csv.writer(my_csv, delimiter=',')
                dummy = []
                for param in self.policy.parameters():
                    dummy1 = param.data.cpu().detach().numpy()
                    for p1 in dummy1:
                        if(np.isscalar(p1)):
                        
                            dummy.append(p1)
                        else:
                            for p2 in p1:
                            
                                dummy.append(p2)
                csvWriter.writerow(dummy)
            with open("critic_params.csv", "a", newline = '') as my_csv:
                csvWriter = csv.writer(my_csv, delimiter=',')
                dummy = []
                for param in self.critic.parameters():
                    dummy1 = param.data.cpu().detach().numpy()
                    for p1 in dummy1:
                        if(np.isscalar(p1)):
                        
                            dummy.append(p1)
                        else:
                            for p2 in p1:
                            
                                dummy.append(p2)
                csvWriter.writerow(dummy)


class FixedHebbRecurrentAgent(object):
    def __init__(self, lr, isize, hsizes, num_actions,act_functions, rec_layer_out, rec_layer_in, cuda, gamma = 0.99,  eps = 1e-4, weight_decay = 0, blossv = 0.1, batch_size = 1, save_path = "",model_save_name = "ACmodel_hebb"  ):
        # lr (float []) = learning rate [0]- policy, [1] - critic
        # isize (int) = input size
        # hsizes (int []) = sizes of the hidden layers
        # num_actions (int) = output size (number of actions)
        # act_functions (int []) - activation function for each layer. 0 in None (generally used for last layer), 1 is TANH, 2 is SIGMOID, 3 is RELU
        # rec_layer_out (int) - The layer where the recurrent values go out (I.E. 0 is after the first layer activation function)
        # rec_layer_in (int) - The layer where the recurrent values come in (I.E. 0 is before the first layer activation function)
        # cuda {"cuda" or "cpu")
        # gamma (float) - discounting factor for rewards
        # eps (float) - epsilon value for Adam optimizer - DEFAULT: 1e-4
        # weight decay (float) - weight decay for Adam optimizer - DEFAULT: 0
        # blossv (float) - coefficient for value prediction loss - DEFAULT: 0.1
        # batch_size (int) - the batch size the network - DEFAULT: 1
        # save_path (string) - the path to save to  - DEFAULT: ""
        # model_save_name (string) - the name of the model - DEFAULT: ACmodel_hebb
        self.gamma = gamma
        self.cuda = cuda
        self.bs = batch_size
        self.blossv = blossv
        self.model_save_name = model_save_name
        self.save_path = save_path
        self.writeWeights = False;

        self.policy = FixedHebbianNetwork(lr[0], isize, hsizes, num_actions,act_functions,rec_layer_out, rec_layer_in, cuda,eps , weight_decay , batch_size)
        self.critic = FixedHebbianNetwork(lr[1], isize, hsizes, 1,act_functions,rec_layer_out, rec_layer_in, cuda,eps, weight_decay, batch_size)
        if(self.writeWeights):
            with open("policy_params.csv", "w+", newline = '') as my_csv:
                csvWriter = csv.writer(my_csv, delimiter=',')
                dummy = []
                for param in self.policy.parameters():
                    dummy1 = param.data.cpu().detach().numpy()
                    for p1 in dummy1:
                        if(np.isscalar(p1)):
                    
                            dummy.append(p1)
                        else:
                            for p2 in p1:
                        
                                dummy.append(p2)
                csvWriter.writerow(dummy)
            with open("critic_params.csv", "w+", newline = '') as my_csv:
                csvWriter = csv.writer(my_csv, delimiter=',')
                dummy = []
                for param in self.critic.parameters():
                    dummy1 = param.data.cpu().detach().numpy()
                    for p1 in dummy1:
                        if(np.isscalar(p1)):
                        
                            dummy.append(p1)
                        else:
                            for p2 in p1:
                            
                                dummy.append(p2)
                csvWriter.writerow(dummy)
        print(self.policy)
        print(self.critic)
        self.zero_loss()

    def zero_loss(self):
        self.actor_loss = torch.tensor([0.]).to(self.cuda);
        self.critic_loss =  torch.tensor([0.]).to(self.cuda);
        self.R = torch.zeros(self.bs).to(self.cuda)
        self.rewards = []
        self.vs = []
        self.loss = 0
        self.lossv = 0
        self.log_probs = []
        self.critic.optimizer.zero_grad()
        self.policy.optimizer.zero_grad()

        self.critic_rec =  self.critic.initialZeroState().to(self.cuda)
        self.critic_rec_hebb = self.critic.initialZeroHebb().to(self.cuda)
        self.policy_rec = self.policy.initialZeroState().to(self.cuda)
        self.policy_rec_hebb = self.policy.initialZeroHebb().to(self.cuda)


    def store_rewards(self, reward):

        #Get the critics thought at the last state
        #[critic_value, self.critic_rec, self.critic_rec_hebb] = self.critic.forward(state, self.critic_rec, self.critic_rec_hebb)
        #self.vs.append(critic_value)
        self.rewards.append(reward.copy())

    def choose_action(self, observation):
        #[probabilties,self.policy_rec, self.policy_rec_hebb, v] = self.policy(observation, self.policy_rec,self.policy_rec_hebb)
        #print("policy rec:", self.policy_rec)
        #print("policy rec hebb:", self.policy_rec_hebb)
        probabilties, self.policy_rec, self.policy_rec_hebb = self.policy(torch.from_numpy(observation).to(self.cuda), self.policy_rec, self.policy_rec_hebb)

        v, self.critic_rec, self.critic_rec_hebb = self.critic(torch.from_numpy(observation).to(self.cuda), self.critic_rec, self.critic_rec_hebb)

        #print("before",probabilties)
        probabilties = torch.softmax(probabilties, dim=1)
        #print("after", probabilties)
        self.loss += (0.03 * probabilties.pow(2).sum() / self.bs)
        self.vs.append(v)
        action_probs = torch.distributions.Categorical(probabilties)

        action = action_probs.sample();
        self.log_probs.append(action_probs.log_prob(action))
        return action.cpu().numpy()

    def learn(self):
        eplen = len(self.rewards)
        self.R = torch.zeros(self.bs).to(self.cuda)

        for numstepb in reversed(range(eplen)):
            self.R = self.gamma * self.R + torch.from_numpy(self.rewards[numstepb]).to(self.cuda)

            ctrR = self.R - self.vs[numstepb][0]
            self.lossv += ctrR.pow(2).sum() / self.bs
            self.loss -= (self.log_probs[numstepb] * ctrR.detach()).sum() / self.bs

        self.loss += self.blossv * self.lossv
        self.loss /= eplen
        self.loss.backward()
        self.critic.optimizer.step()
        self.policy.optimizer.step()
        self.save_params()
        return self.loss
    def save_model(self):
        torch.save(self.policy.state_dict(), self.save_path+"policy_"+self.model_save_name)
        torch.save(self.critic.state_dict(), self.save_path+"critic_"+self.model_save_name)
    def load_model(self, model_path_policy, model_path_critic):
        self.policy.load_state_dict(torch.load(model_path_policy))
        self.critic.load_state_dict(torch.load(model_path_critic))
    def save_params(self):
        if(self.writeWeights):
            with open("policy_params.csv", "a", newline = '') as my_csv:
                csvWriter = csv.writer(my_csv, delimiter=',')
                dummy = []
                for param in self.policy.parameters():
                    dummy1 = param.data.cpu().detach().numpy()
                    for p1 in dummy1:
                        if(np.isscalar(p1)):
                        
                            dummy.append(p1)
                        else:
                            for p2 in p1:
                            
                                dummy.append(p2)
                csvWriter.writerow(dummy)
            with open("critic_params.csv", "a", newline = '') as my_csv:
                csvWriter = csv.writer(my_csv, delimiter=',')
                dummy = []
                for param in self.critic.parameters():
                    dummy1 = param.data.cpu().detach().numpy()
                    for p1 in dummy1:
                        if(np.isscalar(p1)):
                        
                            dummy.append(p1)
                        else:
                            for p2 in p1:
                            
                                dummy.append(p2)
                csvWriter.writerow(dummy)

