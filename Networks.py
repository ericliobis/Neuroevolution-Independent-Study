#import argparse
#import pdb
#from line_profiler import LineProfiler
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from enum import Enum
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


#Activation Functions
def tanh_act(hidden):
    return torch.tanh(hidden)
def sig_act(hidden):
    return torch.sigmoid(hidden)
def relu_act(hidden):
    return torch.relu(hidden)
def no_act(hidden):
    return hidden

act_functions = {
    0: no_act,
    1: tanh_act,
    2: sig_act,
    3: relu_act
}

def act_layer(function, hidden):
    func = act_functions.get(function, "error")
    return func(hidden)



###############################
#Regular Feed forward Network#
##############################

class RegularNetwork(nn.Module):

    def __init__(self, lr, isize, hsizes, num_actions,act_functions, cuda,eps = 1e-4, weight_decay = 0,  batch_size = 1):
        # lr (float) = learning rate
        # isize (int) = input size
        # hsizes (int []) = sizes of the hidden layers
        # num_actions (int) = output size (number of actions)
        # act_functions (int []) - activation function for each layer. 0 in None (generally used for last layer), 1 is TANH, 2 is SIGMOID, 3 is RELU
        # cuda {"cuda" or "cpu")
        # eps (float) - epsilon value for Adam optimizer - DEFAULT: 1e-4
        # weight decay - weight decay for Adam optimizer - DEFAULT: 0


        super(RegularNetwork, self).__init__()

        self.isize, self.networksize = isize, len(hsizes)+1
        self.device = cuda
        self.lr = lr
        self.batch_size = batch_size
        self.numactions = num_actions
        self.inputsizes = np.insert(hsizes, 0, isize)
        self.outputsizes = np.append(hsizes, self.numactions)
        self.layers = nn.ModuleList()
        self.act_function_list = act_functions

        #Create the network
        for i in range(self.networksize):
            self.layers.append(torch.nn.Linear(self.inputsizes[i], self.outputsizes[i]))


        self.optimizer = optim.Adam(self.parameters(), lr = self.lr,eps=eps, weight_decay=weight_decay)
        self.to(self.device)


    def forward(self, inputs):
        dummy = self.initialZeroState();

        hidden = inputs;

        for i in range(self.networksize):
            hidden = act_layer(self.act_function_list[i], self.layers[i](hidden));
        hidden = hidden+dummy;

        return hidden
    def initialZeroState(self):
        return Variable(torch.zeros(self.batch_size,self.numactions), requires_grad=False ).to(self.device)


###################
#Reccurent Network#
###################

class RecurrentNetwork(nn.Module):

    def __init__(self, lr, isize, hsizes, num_actions,act_functions,rec_layer_out, rec_layer_in, cuda,eps = 1e-4, weight_decay = 0, batch_size = 1):
        # lr (float) = learning rate
        # isize (int) = input size
        # hsizes (int []) = sizes of the hidden layers
        # num_actions (int) = output size (number of actions)
        # act_functions (int []) - activation function for each layer. 0 in None (generally used for last layer), 1 is TANH, 2 is SIGMOID, 3 is RELU
        # rec_layer_out (int) - The layer where the recurrent values go out (I.E. 0 is after the first layer activation function)
        # rec_layer_in (int) - The layer where the recurrent values come in (I.E. 0 is before the first layer activation function)
        # cuda {"cuda" or "cpu")
        # eps (float) - epsilon value for Adam optimizer - DEFAULT: 1e-4
        # weight decay (float) - weight decay for Adam optimizer - DEFAULT: 0
        # batch_size (int) - the batch size the network - DEFAULT: 1
        super(RecurrentNetwork, self).__init__()
        self.isize, self.networksize = isize, len(hsizes)+1
        self.device = cuda
        self.lr = lr
        self.numactions = num_actions
        self.inputsizes = np.insert(hsizes, 0, isize)
        self.outputsizes = np.append(hsizes, self.numactions)
        self.layers = nn.ModuleList()
        self.act_function_list = act_functions
        self.rec_layer_in = rec_layer_in
        self.rec_layer_out = rec_layer_out
        self.batch_size = batch_size
        self.rec_size = self.inputsizes[rec_layer_out + 1]
        self.w = torch.nn.Parameter(
            .001 * torch.rand(self.rec_size, self.rec_size))

        #Create the network
        for i in range(self.networksize):
            self.layers.append(torch.nn.Linear(self.inputsizes[i], self.outputsizes[i]))

        self.optimizer = optim.Adam(self.parameters(), lr = self.lr,eps=eps, weight_decay=weight_decay)
        self.to(self.device)


    def forward(self, inputs, recurrent_vals):
        hidden = inputs;
        rec_layer_values = self.initialZeroState();
        recurrent_vals = self.initialZeroState();
        for i in range(self.networksize):

            #If we're on the recurrent input layer, add it to the current layer


            hidden = self.layers[i](hidden)
            if i == self.rec_layer_in:

                hidden += recurrent_vals.matmul(self.w)


            hidden = act_layer(self.act_function_list[i],hidden)
            if i == self.rec_layer_out:
                rec_layer_values = hidden

        return hidden, rec_layer_values
    def initialZeroState(self):
        return Variable(torch.zeros(self.batch_size,self.rec_size), requires_grad=False ).to(self.device)

#################
#Hebbian Network#
#################



class HebbianNetwork(nn.Module):

    def __init__(self, lr, isize, hsizes,num_actions ,act_functions,rec_layer_out, rec_layer_in, cuda,eps = 1e-4, weight_decay = 0, batch_size = 1):
        # lr (float) = learning rate
        # isize (int) = input size
        # hsizes (int []) = sizes of the hidden layers
        # num_actions (int) = output size (number of actions)
        # act_functions (int []) - activation function for each layer. 0 in None (generally used for last layer), 1 is TANH, 2 is SIGMOID, 3 is RELU
        # rec_layer_out (int) - The layer where the recurrent values go out (I.E. 0 is after the first layer activation function)
        # rec_layer_int (int) - The layer where the recurrent values come in (I.E. 0 is before the first layer activation function)
        # cuda {"cuda" or "cpu")
        # eps (float) - epsilon value for Adam optimizer - DEFAULT: 1e-4
        # weight decay (float) - weight decay for Adam optimizer - DEFAULT: 0
        # batch_size (int) - the batch size the network - DEFAULT: 1
        super(HebbianNetwork, self).__init__()
        self.isize, self.networksize = isize, len(hsizes)+1
        self.device = cuda
        self.lr = lr
        self.batch_size = batch_size
        self.numactions = num_actions
        self.inputsizes = np.insert(hsizes, 0, isize)
        self.outputsizes = np.append(hsizes, self.numactions)
        self.layers = nn.ModuleList()
        self.act_function_list = act_functions
        self.rec_layer_in = rec_layer_in
        self.rec_layer_out = rec_layer_out
        self.rec_size = self.inputsizes[rec_layer_out+1]

        #Hebbian/differentiable plasticity/Backpropamine weights
        self.w = torch.nn.Parameter(
            .001 * torch.rand(self.rec_size, self.rec_size))
        self.alpha = torch.nn.Parameter(.001 * torch.rand(self.rec_size,
                                                          self.rec_size))

        self.h2mod = torch.nn.Linear(self.rec_size, 1)
        self.modfanout = torch.nn.Linear(1,
                                         self.rec_size)

        #Create the network
        for i in range(self.networksize):
            self.layers.append(torch.nn.Linear(self.inputsizes[i], self.outputsizes[i]))

        self.optimizer = optim.Adam(self.parameters(), lr = self.lr, eps=eps, weight_decay=weight_decay)
        self.to(self.device)


    def forward(self, inputs, recurrent_vals, hebb):
        hidden = inputs
        rec_layer_values = self.initialZeroState();
        #hebb = self.initialZeroHebb();
        #recurrent_vals = self.initialZeroState();

        #hebb_m = Variable(torch.zeros(self.batch_size, self.rec_size, self.rec_size) , requires_grad=False).to(self.device)



        for i in range(self.networksize):
            if i == self.rec_layer_in:

                h_  = self.layers[i](hidden) + recurrent_vals.unsqueeze(1).bmm(self.w + torch.mul(self.alpha, hebb)).squeeze(1)
                hidden = act_layer(self.act_function_list[i], h_)
            else:
                hidden =  act_layer(self.act_function_list[i], (self.layers[i](hidden)))
            if i == self.rec_layer_out:

                rec_layer_values = hidden


        myeta = F.tanh(self.h2mod(rec_layer_values)).unsqueeze(2)
        deltahebb = torch.bmm(recurrent_vals.unsqueeze(2), rec_layer_values.unsqueeze(1))
        myeta = self.modfanout(myeta)

        self.clipval = 2
        hebb_return = torch.clamp(hebb + deltahebb, min=-self.clipval, max=self.clipval)
        hebb_return = torch.clamp(hebb + myeta * deltahebb, min=-self.clipval, max=self.clipval)

        return hidden, rec_layer_values, hebb_return

    def initialZeroState(self):
        return Variable(torch.zeros(self.batch_size,self.rec_size), requires_grad=False ).to(self.device)
    def initialZeroHebb(self):
        return Variable(torch.zeros(self.batch_size, self.rec_size, self.rec_size) , requires_grad=False).to(self.device)


#####################
#A/C Hebbian Network#
#####################



class ACHebbianNetwork(nn.Module):

    def __init__(self, lr, isize, hsizes,num_actions ,act_functions,rec_layer_out, rec_layer_in, cuda,eps = 1e-4, weight_decay = 0, batch_size = 1):
        # lr (float) = learning rate
        # isize (int) = input size
        # hsizes (int []) = sizes of the hidden layers
        # num_actions (int) = output size (number of actions)
        # act_functions (int []) - activation function for each layer. 0 in None (generally used for last layer), 1 is TANH, 2 is SIGMOID, 3 is RELU
        # rec_layer_out (int) - The layer where the recurrent values go out (I.E. 0 is after the first layer activation function)
        # rec_layer_int (int) - The layer where the recurrent values come in (I.E. 0 is before the first layer activation function)
        # cuda {"cuda" or "cpu")
        # eps (float) - epsilon value for Adam optimizer - DEFAULT: 1e-4
        # weight decay (float) - weight decay for Adam optimizer - DEFAULT: 0
        # batch_size (int) - the batch size the network - DEFAULT: 1
        super(ACHebbianNetwork, self).__init__()
        self.isize, self.networksize = isize, len(hsizes)+1
        self.device = cuda
        self.lr = lr
        self.batch_size = batch_size
        self.numactions = num_actions
        self.inputsizes = np.insert(hsizes, 0, isize)
        self.outputsizes = np.append(hsizes, self.numactions)
        self.layers = nn.ModuleList()
        self.act_function_list = act_functions
        self.rec_layer_in = rec_layer_in
        self.rec_layer_out = rec_layer_out
        self.rec_size = self.inputsizes[rec_layer_out+1]
        self.critic_output = torch.nn.Linear(hsizes[-1],1)

        #Hebbian/differentiable plasticity/Backpropamine weights
        self.w = torch.nn.Parameter(
            .001 * torch.rand(self.rec_size, self.rec_size))
        self.alpha = torch.nn.Parameter(.001 * torch.rand(self.rec_size,
                                                          self.rec_size))

        self.h2mod = torch.nn.Linear(self.rec_size, 1)
        self.modfanout = torch.nn.Linear(1,
                                         self.rec_size)

        #Create the network
        for i in range(self.networksize):
            self.layers.append(torch.nn.Linear(self.inputsizes[i], self.outputsizes[i]))

        self.optimizer = optim.Adam(self.parameters(), lr = self.lr, eps=eps, weight_decay=weight_decay)
        self.to(self.device)


    def forward(self, inputs, recurrent_vals, hebb):
        hidden = inputs
        rec_layer_values = self.initialZeroState();
        criticVal = [0];
        #hebb = self.initialZeroHebb();
        #recurrent_vals = self.initialZeroState();

        hebb_m = Variable(torch.zeros(self.batch_size, self.rec_size, self.rec_size) , requires_grad=False).to(self.device)



        for i in range(self.networksize):
            if (i == self.networksize):
                criticVal = [self.critic_output(hidden)]
            if i == self.rec_layer_in:

                h_  = self.layers[i](hidden) + recurrent_vals.unsqueeze(1).bmm(self.w + torch.mul(self.alpha, hebb)*hebb_m).squeeze(1)
                hidden = act_layer(self.act_function_list[i], h_)
            else:
                hidden =  act_layer(self.act_function_list[i], (self.layers[i](hidden)))
            if i == self.rec_layer_out:

                rec_layer_values = hidden


        myeta = F.tanh(self.h2mod(rec_layer_values)).unsqueeze(2)
        deltahebb = torch.bmm(recurrent_vals.unsqueeze(2), rec_layer_values.unsqueeze(1))
        myeta = self.modfanout(myeta)

        self.clipval = 2
        hebb_return = torch.clamp(hebb + deltahebb, min=-self.clipval, max=self.clipval)
        hebb_return = torch.clamp(hebb + myeta * deltahebb, min=-self.clipval, max=self.clipval)

        return hidden, rec_layer_values, hebb_return, criticVal

    def initialZeroState(self):
        return Variable(torch.zeros(self.batch_size,self.rec_size), requires_grad=False ).to(self.device)
    def initialZeroHebb(self):
        return Variable(torch.zeros(self.batch_size, self.rec_size, self.rec_size) , requires_grad=False).to(self.device)


