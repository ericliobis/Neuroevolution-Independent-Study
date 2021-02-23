import argparse
import pdb
#from line_profiler import LineProfiler
import torch
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
class HebbianNetwork(nn.Module):

    def __init__(self, isize, hsize):
        super(HebbianNetwork, self).__init__()
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
        hactiv = torch.tanh(self.i2h(inputs) + hidden[0].unsqueeze(1).bmm(self.w + torch.mul(self.alpha, hebb)).squeeze(
            1))  # Update the h-state
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

    def initialZeroState(self, BATCHSIZE):
        return Variable(torch.zeros(BATCHSIZE, self.hsize), requires_grad=False)

    # In plastic networks, we must also initialize the Hebbian state:
    def initialZeroHebb(self, BATCHSIZE):
        return Variable(torch.zeros(BATCHSIZE, self.hsize, self.hsize), requires_grad=False)

# That's it for plasticity! The rest of the code simply implements the maze task and the A2C RL algorithm.
