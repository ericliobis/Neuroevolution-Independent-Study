import numpy as np
import random
import torch
import gym
from os import path
#from Agents import PolicyAgent
#from Agents import ACAgent
#from Agents import ACRecurrentAgent
#from Agents import ACHebbRecurrentAgent
#from pole_environment import CartPoleEnv
#from pole_environment import CartPoleEnv_Rand_Length
from utils import plotLearning
import argparse
import pickle
import time
from PIL import Image
import imageio
import os
import platform
import csv
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#import matplotlib.colors as colors
#from matplotlib import cm
from numpy import genfromtxt
from matplotlib.animation import FuncAnimation


save_name = "AC_Target_AC_5"
save_path = "./SavedNetworks/"
data_path = "./SavedData/"


parser = argparse.ArgumentParser()
parser.add_argument("file_loc", help="The Location of the CSV to plot")
args = parser.parse_args()


def load_and_plot(thing_to_plot,every, anim):
    plot_name = ""
    if(thing_to_plot ==0):
        plot_name = "DIST_HIST_"
    if (thing_to_plot == 1):
        plot_name = "SCORE_HIST_"
    if (thing_to_plot == 2):
        plot_name = "BALANCE_HIST_"
    test = []
    line1 = []
    with open(args.file_loc, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            test.append(line)

    n = int(len(test)/every)
    colors = plt.cm.RdYlGn(np.linspace(0,1,n))

    fig = plt.figure()
    ax = plt.axes(xlim=(0, 1000), ylim=(0, 1))
    line, = ax.plot([], [], lw=3)
    def init():
        line.set_data([], [])
        return line,

    def animate(i):

        y = line1[i]
        fig.suptitle(int(i*every))
        x = np.arange(len(y))
        line.set_data(x, y)
        return line,

    if anim:

        for i in range(n):
            y1 = [float(j) for j in test[i*every]]

            line1.append(y1)

        anim = FuncAnimation(fig, animate, init_func=init,
                             frames=100, interval=300, blit=True)
        anim.save(args.file_loc+'.gif', writer='imagemagick')


    else:
        for i in range(n):

            line = [float(j) for j in test[i*every]]
            print(line)
            plt.plot(line, color = colors[i])
            plt.xlabel('Time')
            plt.ylabel('test')
        plt.savefig(args.file_loc+'.png')

load_and_plot(2,20,True)