import numpy as np
import random
import torch
#import gym
from os import path
from Agents import PolicyAgent
from Agents import ACAgent
from Agents import ACRecurrentAgent
from Agents import ACHebbRecurrentAgent
from Agents import ACHebbRecurrentAgent_
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
#import matplotlib.pyplot as plt
#import matplotlib.colors as colors
#from matplotlib import cm
#from numpy import genfromtxt


config = configparser.ConfigParser()
config.read('config.ini')
seed =  int(config['DEFAULT']['seed'])
multF  = (config['DEFAULT']['multF'] == "True")
num_runs =  int(config['DEFAULT']['number_of_runs'])
act_fun = [int(x) for x in config.get('NETWORK_ACT', 'act_fun').split(',')]
hsize = [int(x) for x in config.get('NETWORK_SHAPE', 'hsize').split(',')]
##Create the environment
env = CartPoleEnv_Rand_Length(seed, multF)


#The number of inputs. Currently the number of observations plus last reward and last action
isize =env.observation_space.shape[0] + 2;
num_actions = env.action_space.n


#Network characteristic
learning_rate = float(config['NETWORK_PARAMS']['learning_rate'])
learning_rate_policy = float(config['NETWORK_PARAMS']['learning_rate_policy'])
learning_rate_critic = float(config['NETWORK_PARAMS']['learning_rate_critic'])
rec_layer_in = int(config['NETWORK_PARAMS']['rec_layer_in'])
rec_layer_out = int(config['NETWORK_PARAMS']['rec_layer_out'])
batch_size = int(config['NETWORK_PARAMS']['batch_size'])
gamma = float(config['NETWORK_PARAMS']['gamma'])
blossv = float(config['NETWORK_PARAMS']['blossv'])
eps = float(config['NETWORK_PARAMS']['eps'])
weight_decay= float(config['NETWORK_PARAMS']['weight_decay'])

#Network to use
## 0 = Policy Gradient
## 1 = Actor Critic
## 2 = Actor Critic with Reccurent
## 3 = Actor Critic with Hebbian
network = int(config['NETWORK_PARAMS']['network'])

#Other Network characteristics
save_every = int(config['EPISODE']['save_every'])
max_episodes = int(config['EPISODE']['max_episodes'])
length_of_episodes = int(config['EPISODE']['length_of_episodes'])


#The name of the network will be saved and it's location
save_name = "NET_" + str(network)+ "_LR_" + str(learning_rate) + "_LRC_" + str(learning_rate_critic) + "_LRP_" + str(learning_rate_policy) +"_NAME_"+ config['SAVING']['save_name']
data_path = config['SAVING']['data_path']

if not os.path.exists(data_path):
    os.makedirs(data_path)

save_path = data_path+str(save_name)+"/"

if not os.path.exists(save_path):
    os.makedirs(save_path)

copyfile('config.ini',save_path + 'config.ini' )

render  = (config['EPISODE']['render'] == "True")
learn  = (config['NETWORK_PARAMS']['learn'] == "True")
render_eps = int(config['EPISODE']['render_eps'])
target_range_lower = float(config['EPISODE']['target_range_lower'])
target_range_upper = float(config['EPISODE']['target_range_upper'])

#sets the seed
np.random.seed(seed);
random.seed(seed);
torch.manual_seed(seed)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")





def runNetworks(learn, network_type, network_save_name, save_eps,  max_ep, episode_length, render, render_eps, target_range_lower, target_range_upper, number_of_runs):
    #learn (boolean) - do you want this network to learn or not
    #network_type (int) - Network to use, 0 = Policy Gradient, 1 = Actor Critic, 2 = Actor Critic with Reccurent, 3 = Actor Critic with Hebbian
    #network_save_name (string) - name of the save file for the network
    #save_eps (int) - how often to save the network
    #max_ep (int) - max number of episodes to run
    #episode_length (int) - number of frames to run each episode
    #render (boolean) - whether to render episodes or not
    #render_eps (int) - how often to render. 1 is every episode, 10 would be every 10 episodes, etc
    #target_range_lower (float) - the lower value of where the target can be
    #target_range_upper (float) - the upper value of where the target can be
    #number_of_runs (int) - number of runs to do.

    score_history_all = []
    #Create the right network depending on the type
    for run in range(number_of_runs):
        dist_history_all = []
        balance_history_all = []
        score_history_round = []

        score_history = []
        max_episodes = max_ep
        ep_len = episode_length
        save = []
        save_path_temp = save_path+"/"
        network_save_name_temp = "run_"+str(run)


        if network_type == 0:
            agent = PolicyAgent([learning_rate], isize, hsize, num_actions, act_fun, device, gamma, eps, 0, batch_size, save_path_temp,network_save_name_temp)
            save.append(save_path_temp + "Regular_agent_" + network_save_name_temp);
        else:
            save.append(save_path_temp + "policy_" + network_save_name_temp)
            save.append(save_path_temp + "critic_" + network_save_name_temp)
            if network_type == 1:
                agent = ACAgent([learning_rate_policy, learning_rate_critic], isize, hsize, num_actions,  act_fun, device, gamma, eps, weight_decay, blossv, batch_size, save_path_temp,network_save_name_temp)
            elif network_type == 2:
                agent = ACRecurrentAgent([learning_rate_policy, learning_rate_critic], isize, hsize, num_actions,  act_fun, rec_layer_out, rec_layer_in,device, gamma, eps, weight_decay, blossv, batch_size, save_path_temp,network_save_name_temp)
            elif network_type == 3:
                agent = ACHebbRecurrentAgent([learning_rate_policy, learning_rate_critic], isize, hsize, num_actions,  act_fun, rec_layer_out, rec_layer_in,device, gamma, eps, weight_decay, blossv, batch_size, save_path_temp,network_save_name_temp)
            elif network_type == 4:
                agent = ACHebbRecurrentAgent_([learning_rate_policy, learning_rate_critic], isize, hsize, num_actions,  act_fun, rec_layer_out, rec_layer_in,device, gamma, eps, weight_decay, blossv, batch_size, save_path_temp,network_save_name_temp)


        #Check to see if the network already exists and load it if it does
        if network_type == 0 and path.exists(save[0]):

            agent.load_model(save[0])
        elif (path.exists(save[0]) and  path.exists(save[1])):
            agent.load_model(save[0],save[1])



        for i_episode in range(max_episodes):
            dist_hist = []
            score_hist = []
            balance_hist = []

            frames = [];
            env.setTarget(random.uniform(target_range_lower, target_range_upper))
            last_reward = 0.0
            last_action = 0.0;
            agent.zero_loss()
            score = 0
            observation_temp = env.reset()
            observation = observation_temp


            for t in range(ep_len):


                if render and i_episode % render_eps == 0:
                    env.render()


                #Take the observation and append the extra values
                obvs = np.array([np.append(np.asarray([observation], dtype=np.float32), np.asarray([last_action,last_reward], dtype=np.float32))])

                #choose an action
                action = agent.choose_action(obvs);



                observation_temp, reward, done, info = env.step(action[0])
                dist_hist.append(env.getDist())
                balance_hist.append(env.getPercentBal())
                score_hist.append(reward)

                observation_ = observation_temp

                #Save the last action and reward
                last_action = action[0]
                last_reward = reward


                ######RECORDS AND SAVES THE STATE#####
                agent.store_rewards(np.array(reward))


                #######################################
                score += reward

                observation = observation_

                #if it falls over or the episode ends, we break
                if done or t == ep_len-1:

                    #add the score to the history
                    score_history.append(score)

                    #If we've get it to learn, then we learn.
                    if(learn):
                        agent.learn()


                    print("==========================================")
                    print("Episode: ", i_episode)
                    print("Reward: ", score)
                    print("Length:", env.length)

                    break
            #dist_history_all.append(dist_hist);
            #balance_history_all.append(balance_hist);
            score_history_round.append(sum(score_hist))
            if i_episode%save_eps == 0:
                agent.save_model()
                print("Model Saved")

        #plotLearning(score_history, filename = "cartpole.png", window = 10)
        env.close()
        score_history_all.append(score_history_round)
        #with open(save_path_temp+"DIST_HIST_"+save_name+ ".csv", "w+", newline = '') as my_csv:
        #    csvWriter = csv.writer(my_csv, delimiter=',')
        #    csvWriter.writerows(dist_history_all)
    with open(save_path_temp+"SCORE_HIST_ALL.csv", "w+", newline = '') as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(score_history_all)
        #with open(save_path_temp+"BALANCE_HIST_"+save_name+ ".csv", "w+", newline = '') as my_csv:
        #    csvWriter = csv.writer(my_csv, delimiter=',')
        #    csvWriter.writerows(balance_history_all)


runNetworks(learn, network, save_name, save_every, max_episodes,length_of_episodes, render, render_eps, target_range_lower, target_range_upper, num_runs)

#load_and_plot(0)