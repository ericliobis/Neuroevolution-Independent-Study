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

    def __init__(self, lr, isize, hsizes, NBACTIONS,cuda, sf, rec_layer_out, rec_layer_in, bs):
        super(HebbianNetwork, self).__init__()
        self.isize, self.networksize = isize, len(hsizes)+1
        self.device = cuda
        self.lr = lr
        self.batchSize = bs
        #the input sizes for each part of the network
        self.numactions = NBACTIONS
        self.inputsizes = np.insert(hsizes, 0, isize)
        self.outputsizes = np.append(hsizes, self.numactions)
        self.layers = nn.ModuleList()
        self.sf = sf
        self.rec_layer_in = rec_layer_in
        self.rec_layer_out = rec_layer_out
        self.rec_size = self.inputsizes[rec_layer_out+1]

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

        self.optimizer = optim.Adam(self.parameters(), lr = self.lr, eps=1e-4, weight_decay=0.0)
        self.to(self.device)


    def forward(self, inputs, recurrent_vals, hebb):
        state = inputs#torch.Tensor(inputs).to(self.device)

        hactiv = state;
        rec_layer_values = self.initialZeroState();
        for i in range(self.networksize-1):
            if i == self.rec_layer_in:
                hactiv = torch.tanh(self.layers[i](hactiv) + recurrent_vals.unsqueeze(1).bmm(self.w + torch.mul(self.alpha, hebb)).squeeze(1))
            else:
                hactiv = torch.tanh(self.layers[i](hactiv))
            if i == self.rec_layer_out:
                rec_layer_values = hactiv

        hactiv = (self.layers[self.networksize-1](hactiv))

        if self.sf:
            act_dis = torch.softmax(hactiv, dim=1)
        else:
            act_dis = hactiv

        myeta = F.tanh(self.h2mod(rec_layer_values)).unsqueeze(2)

        deltahebb = torch.bmm(recurrent_vals.unsqueeze(2), rec_layer_values.unsqueeze(1))
        myeta = self.modfanout(myeta)

        self.clipval = 2.0
        hebb_return = torch.clamp(hebb + myeta * deltahebb, min=-self.clipval, max=self.clipval)





        return act_dis, rec_layer_values, hebb_return

    def initialZeroState(self):
        return Variable(torch.zeros(self.batchSize,self.rec_size), requires_grad=False ).to(self.device)
    def initialZeroHebb(self):
        return Variable(torch.zeros(self.batchSize, self.rec_size, self.rec_size) , requires_grad=False).to(self.device)

    def set_network_params(self,w_i, alpha_i,h2mod_i,modfanout_i, layer_1, layer_2):
        self.w = w_i
        self.alpha = alpha_i
        self.h2mod = h2mod_i
        self.modfanout = modfanout_i
        self.layers[0] = layer_1;
        self.layers[1] = layer_2;




class HebbianNetworkTEST(nn.Module):

    def __init__(self, isize, hsize):

        super(HebbianNetworkTEST, self).__init__()
        NBACTIONS = 4;
        self.hsize, self.isize = hsize, isize

        self.i2h = torch.nn.Linear(isize, hsize)  # Weights from input to recurrent layer
        self.w = torch.nn.Parameter(
            .001 * torch.rand(hsize, hsize))  # Baseline (non-plastic) component of the plastic recurrent layer

        self.alpha = torch.nn.Parameter(.001 * torch.rand(hsize,
                                                          hsize))  # Plasticity coefficients of the plastic recurrent layer; one alpha coefficient per recurrent connection
        # self.alpha = torch.nn.Parameter(.0001 * torch.rand(1,1,hsize))  # Per-neuron alpha
        # self.alpha = torch.nn.Parameter(.0001 * torch.ones(1))         # Single alpha for whole network

        self.h2mod = torch.nn.Linear(hsize, 1)  # Weights from the recurrent layer to the (single) neurodulator output
        self.modfanout = torch.nn.Linear(1,
                                         hsize)  # The modulator output is passed through a different 'weight' for each neuron (it 'fans out' over neurons)

        self.h2o = torch.nn.Linear(hsize, NBACTIONS)  # From recurrent to outputs (action probabilities)
        self.h2v = torch.nn.Linear(hsize, 1)  # From recurrent to value-prediction (used for A2C)

    def forward(self, inputs,
                hidden):  # hidden is a tuple containing the h-state (i.e. the recurrent hidden state) and the hebbian trace

        HS = self.hsize

        # hidden[0] is the h-state; hidden[1] is the Hebbian trace
        hebb = hidden[1]

        # Each *column* of w, alpha and hebb contains the inputs weights to a single neuron
        hactiv = torch.tanh(self.i2h(inputs) + hidden[0].unsqueeze(1).bmm(self.w + torch.mul(self.alpha, hebb)).squeeze(            1))  # Update the h-state
        activout = self.h2o(hactiv)  # Pure linear, raw scores - to be softmaxed later, outside the function
        valueout = self.h2v(hactiv)

        # Now computing the Hebbian updates...
        deltahebb = torch.bmm(hidden[0].unsqueeze(2), hactiv.unsqueeze(
            1))  # Batched outer product of previous hidden state with new hidden state

        # We also need to compute the eta (the plasticity rate), wich is determined by neuromodulation
        # Note that this is "simple" neuromodulation.
        myeta = F.tanh(self.h2mod(hactiv)).unsqueeze(2)  # Shape: BatchSize x 1 x 1

        # The neuromodulated eta is passed through a vector of fanout weights, one per neuron.
        # Each *column* in w, hebb and alpha constitutes the inputs to a single cell.
        # For w and alpha, columns are 2nd dimension (i.e. dim 1); for hebb, it's dimension 2 (dimension 0 is batch)
        # The output of the following line has shape BatchSize x 1 x NHidden, i.e. 1 line and NHidden columns for each
        # batch element. When multiplying by hebb (BatchSize x NHidden x NHidden), broadcasting will provide a different
        # value for each cell but the same value for all inputs of a cell, as required by fanout concept.
        myeta = self.modfanout(myeta)

        # Updating Hebbian traces, with a hard clip (other choices are possible)
        self.clipval = 2.0
        hebb = torch.clamp(hebb + myeta * deltahebb, min=-self.clipval, max=self.clipval)

        hidden = (hactiv, hebb)
        return activout, valueout, hidden

    def initialZeroState(self, BATCHSIZE=30):
        return Variable(torch.zeros(BATCHSIZE, self.hsize), requires_grad=False)

    # In plastic networks, we must also initialize the Hebbian state:
    def initialZeroHebb(self, BATCHSIZE=30):
        return Variable(torch.zeros(BATCHSIZE, self.hsize, self.hsize), requires_grad=False)


####LEGACY

class HebbianNetworkForMaze(nn.Module):

    def __init__(self, lr, isize, hsizes, NBACTIONS,cuda, sf, rec_layer_out, rec_layer_in, bs):
        super(HebbianNetworkForMaze, self).__init__()
        self.isize, self.networksize = isize, len(hsizes)+1
        self.device = cuda
        self.lr = lr
        self.batchSize = bs
        #the input sizes for each part of the network
        self.numactions = NBACTIONS
        self.inputsizes = np.insert(hsizes, 0, isize)
        self.outputsizes = np.append(hsizes, self.numactions)
        self.layers = nn.ModuleList()
        self.sf = sf
        self.rec_layer_in = rec_layer_in
        self.rec_layer_out = rec_layer_out
        self.rec_size = self.inputsizes[rec_layer_out+1]
        self.h2v = torch.nn.Linear(self.rec_size, 1)

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

        self.optimizer = optim.Adam(self.parameters(), lr = self.lr, eps=1e-4, weight_decay=0.0)
        self.to(self.device)


    def forward(self, inputs, recurrent_vals, hebb):
        state = torch.Tensor(inputs).to(self.device)
        hactiv = state;
        rec_layer_values = self.initialZeroState();
        for i in range(self.networksize-1):
            if i == self.rec_layer_in:
                hactiv = torch.tanh(self.layers[i](hactiv) + recurrent_vals.unsqueeze(1).bmm(self.w + torch.mul(self.alpha, hebb)).squeeze(1))
            else:
                hactiv = torch.tanh(self.layers[i](hactiv))
            if i == self.rec_layer_out:
                rec_layer_values = hactiv
        v_out = self.h2v(hactiv)
        hactiv = (self.layers[self.networksize-1](hactiv))

        if self.sf:
            act_dis = torch.softmax(hactiv, dim=1)
        else:
            act_dis = hactiv


        myeta = F.tanh(self.h2mod(rec_layer_values)).unsqueeze(2)

        deltahebb = torch.bmm(recurrent_vals.unsqueeze(2), rec_layer_values.unsqueeze(1))
        myeta = self.modfanout(myeta)

        self.clipval = 2.0
        hebb_return = torch.clamp(hebb + myeta * deltahebb, min=-self.clipval, max=self.clipval)
        return act_dis, rec_layer_values, hebb_return, v_out

    def initialZeroState(self):
        return Variable(torch.zeros(self.batchSize,self.rec_size), requires_grad=False ).to(self.device)
    def initialZeroHebb(self):
        return Variable(torch.zeros(self.batchSize, self.rec_size, self.rec_size) , requires_grad=False).to(self.device)

    def set_network_params(self,w_i, alpha_i,h2mod_i,modfanout_i, layer_1, layer_2):
        self.w = w_i
        self.alpha = alpha_i
        self.h2mod = h2mod_i
        self.modfanout = modfanout_i
        self.layers[0] = layer_1;
        self.layers[1] = layer_2;
