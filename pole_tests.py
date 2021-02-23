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
env = CartPoleEnv_Rand_Length(1)
agent4_name_critic = "./SavedData/NET_2_LR_0.01_LRC_0.01_LRP_0.01_NAME_rec_l/critic_NET_2_LR_0.01_LRC_0.01_LRP_0.01_NAME_rec_l"
agent4_name_policy = "./SavedData/NET_2_LR_0.01_LRC_0.01_LRP_0.01_NAME_rec_l/policy_NET_2_LR_0.01_LRC_0.01_LRP_0.01_NAME_rec_l"
def ACHebbAgent_test(learn, record):
    score_history = []
    continue_history = []
    start = 3
    stop = 3
    incr = 0.5
    ep_len = 1000

    starts = 1
    continues = 1



    range1 = np.arange(start, stop, incr)
    range1 = [3];
    print("Range: ", range1)


    for i_targ in np.nditer(range1):
        print("Target: ", i_targ)

        env.setTarget(i_targ)
        continue_history = np.zeros(continues)

        for i_starts in range(starts):
            frames = [];

            #reload the model at each start

            if (path.exists(agent4_name_policy) and path.exists(agent4_name_critic)):
                agent4.load_model(agent4_name_policy, agent4_name_critic)

            for i_episode in range(continues):
                agent4.zero_loss()

                #reset the environment
                last_reward = 0.0
                last_action = 0.0;
                score = 0
                observation_temp = env.reset()
                observation = observation_temp;

                for t in range(ep_len):
                    #if i_episode % 1000 == 0:
                        #env.render()
                    ##First run through of the network. Take the observation, put it through the network
                    obvs = np.append(np.asarray([observation], dtype=np.float32),
                                     np.asarray([last_action, last_reward], dtype=np.float32))
                    action = agent4.choose_action(obvs);

                    #action = agent4.choose_action(np.asarray([observation], dtype=np.float32))


                    observation_temp, reward, done, info = env.step(action[0])
                    observation_ = observation_temp
                    last_action = action[0]
                    last_reward = reward
                    if(record):
                        frames.append(Image.fromarray(env.render(mode='rgb_array')))



                    ######RECORDS AND SAVES THE STATE#####
                    agent4.store_rewards(np.array(reward))


                    #######################################
                    score += reward

                    observation = observation_
                    if done  or t == ep_len-1:
                        #score_history.append(score)
                        if(record):
                            imageio.mimsave("vid1.gif", frames, format='GIF', fps=60)
                        #with open('vid1.gif', 'wb') as f:  # change the path if necessary
                         #   im = Image.new('RGB', frames[0].size)
                         #   //im.save(f, save_all=True, append_images=frames, duration = 1)
                        if (learn):
                            agent4.learn()


                        print("==========================================")
                        print("Episode: ", i_episode)
                        print("Reward: ", score)
                        print("Length:", env.length)
                        print("Target: ", i_targ)

                        break
                continue_history[i_episode] += score
        score_history = np.append(score_history,continue_history/starts)
    #np.savetxt("Hebb1.csv", score_history, delimiter=",")
ACHebbAgent_test(False, True)