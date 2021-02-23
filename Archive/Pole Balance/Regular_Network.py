# Regular network, no recurrent term

import argparse
import pdb
#from line_profiler import LineProfiler
import torch
import torch.nn as nn
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

class ActFunction(Enum):
    TANH = 1
    SIGMOID = 2
    RELU = 3


import numpy as np
class RegularNetwork(nn.Module):

    def __init__(self, lr, isize, hsizes, NBACTIONS,cuda, sf):
        super(RegularNetwork, self).__init__()
        self.isize, self.networksize = isize, len(hsizes)+1
        self.device = cuda
        self.lr = lr
        #the input sizes for each part of the network
        self.numactions = NBACTIONS
        self.inputsizes = np.insert(hsizes, 0, isize)
        self.outputsizes = np.append(hsizes, self.numactions)
        self.layers = nn.ModuleList()
        self.sf = sf


        self.actfunction = ActFunction.RELU
        for i in range(self.networksize):
            self.layers.append(torch.nn.Linear(self.inputsizes[i], self.outputsizes[i]))

        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)
        self.to(self.device)


    def forward(self, inputs):

        softmaxer = nn.Softmax(dim=0)



        state = torch.Tensor(inputs).to(self.device)

        hactiv = state;

        if self.actfunction == ActFunction.RELU:
            for i in range(self.networksize-1):
                hactiv = torch.tanh(self.layers[i](hactiv))
        else:
            print("error")
        hactiv = (self.layers[self.networksize-1](hactiv))

        if self.sf:
            act_dis = softmaxer(hactiv)
        else:
            act_dis = hactiv

        return act_dis

class RecurrentNetwork(nn.Module):

    def __init__(self, lr, isize, hsizes, NBACTIONS,cuda, sf, rec_layer_out, rec_layer_in):
        super(RecurrentNetwork, self).__init__()
        self.isize, self.networksize = isize, len(hsizes)+1
        self.device = cuda
        self.lr = lr
        #the input sizes for each part of the network
        self.numactions = NBACTIONS
        self.inputsizes = np.insert(hsizes, 0, isize)
        self.outputsizes = np.append(hsizes, self.numactions)
        self.layers = nn.ModuleList()
        self.sf = sf
        self.rec_layer_in = rec_layer_in
        self.rec_layer_out = rec_layer_out
        self.rec_size = self.inputsizes[rec_layer_out]


        self.actfunction = ActFunction.RELU
        for i in range(self.networksize):
            self.layers.append(torch.nn.Linear(self.inputsizes[i], self.outputsizes[i]))
            if i == self.rec_layer_out:
                self.rec_layer = torch.nn.Linear(self.rec_size,self.rec_size)

        self.optimizer = optim.Adam(self.parameters(), lr = self.lr, weight_decay=0.9)
        self.to(self.device)


    def forward(self, inputs, recurrent_vals):

        softmaxer = nn.Softmax(dim=0)



        state = torch.Tensor(inputs).to(self.device)
        hactiv = state;
        rec_layer_values = None#self.initialZeroState();

        if self.actfunction == ActFunction.RELU:
            for i in range(self.networksize-1):
                if i == self.rec_layer_in:
                    hactiv += recurrent_vals
                hactiv = torch.relu(self.layers[i](hactiv))
                if i == self.rec_layer_out:
                    rec_layer_values = self.rec_layer(hactiv)
        else:
            print("error")
        hactiv = (self.layers[self.networksize-1](hactiv))

        if self.sf:
            act_dis = softmaxer(hactiv)
        else:
            act_dis = hactiv


        return act_dis, rec_layer_values

    def initialZeroState(self):
        return Variable(torch.zeros(1,self.rec_size), requires_grad=False ).to(self.device)



class HebbianNetwork(nn.Module):

    def __init__(self, lr, isize, hsizes, NBACTIONS,cuda, sf, rec_layer_out, rec_layer_in):
        super(RecurrentNetwork, self).__init__()
        self.isize, self.networksize = isize, len(hsizes)+1
        self.device = cuda
        self.lr = lr
        #the input sizes for each part of the network
        self.numactions = NBACTIONS
        self.inputsizes = np.insert(hsizes, 0, isize)
        self.outputsizes = np.append(hsizes, self.numactions)
        self.layers = nn.ModuleList()
        self.sf = sf
        self.rec_layer_in = rec_layer_in
        self.rec_layer_out = rec_layer_out
        self.rec_size = self.inputsizes[rec_layer_out]

        self.w = torch.nn.Parameter(
            .001 * torch.rand(self.rec_size, self.rec_size))
        self.alpha = torch.nn.Parameter(.001 * torch.rand(self.rec_size,
                                                          self.rec_size))

        self.h2mod = torch.nn.Linear(self.rec_size, 1)  # Weights from the recurrent layer to the (single) neurodulator output
        self.modfanout = torch.nn.Linear(1,
                                         self.rec_size)


        self.actfunction = ActFunction.RELU
        for i in range(self.networksize):
            self.layers.append(torch.nn.Linear(self.inputsizes[i], self.outputsizes[i]))
            if i == self.rec_layer_out:
                self.rec_layer = torch.nn.Linear(self.rec_size,self.rec_size)

        self.optimizer = optim.Adam(self.parameters(), lr = self.lr, weight_decay=0.9)
        self.to(self.device)


    def forward(self, inputs, recurrent_vals, hebb):




        state = torch.Tensor(inputs).to(self.device)
        hactiv = state;
        rec_layer_values = None#self.initialZeroState();
        for i in range(self.networksize-1):
            if i == self.rec_layer_in:
                hactiv = torch.tanh(self.layers[i](hactiv) + recurrent_vals[0].unsqueeze(1).bmm(self.w + torch.mul(self.alpha, hebb)).squeeze(1))
            else:
                hactiv = torch.tanh(self.layers[i](hactiv))
            if i == self.rec_layer_out:
                rec_layer_values = self.rec_layer(hactiv)
        else:
            print("error")
        hactiv = (self.layers[self.networksize-1](hactiv))

        if self.sf:
            act_dis = softmaxer(hactiv)
        else:
            act_dis = hactiv

        myeta = F.tanh(self.h2mod(rec_layer_values)).unsqueeze(2)

        deltahebb = torch.bmm(recurrent_vals.unsqueeze(2), rec_layer_values.unsqueeze(1))
        myeta = self.modfanout(myeta)

        self.clipval = 2.0
        hebb_return = torch.clamp(hebb + myeta * deltahebb, min=-self.clipval, max=self.clipval)





        return act_dis, rec_layer_values, hebb_return

    def initialZeroState(self):
        return Variable(torch.zeros(1,self.rec_size), requires_grad=False ).to(self.device)