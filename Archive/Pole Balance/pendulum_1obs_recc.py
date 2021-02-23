import numpy as np
import random
import torch
import time
import gym
from Agent_Object import PolicyAgent
from Agent_Object import ACAgent
from Agent_Object import ACRecurrentAgent
from pole_environment import CartPoleEnv
from utils import plotLearning
from mountain_car_env import MountainCarEnv
env = MountainCarEnv()
# Variable Definitions
isize = 2 #env.observation_space.shape[0]
seed = 2
hsize = [256,256]
num_actions = env.action_space.n
model_save_name = "mountaincar_1obs_recc"
learning_rate = 0.01
gamma = 0.99

print(len(hsize))


#sets the seed
np.random.seed(seed);
random.seed(seed);
torch.manual_seed(seed)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


agent1 = PolicyAgent(device, [learning_rate], isize, gamma, num_actions, hsize,model_save_name)
agent2 = ACAgent(device, [0.00001, 0.00005], isize, gamma, num_actions, hsize,  model_save_name)
agent3 = ACRecurrentAgent(device, [0.00001, 0.00005], isize, gamma, num_actions, hsize, 1, model_save_name)


#the current agent
agent = agent2


score_history = []
score = 0
max_episodes = 50000


for i_episode in range(max_episodes):
    score = 0
    observation_temp = env.reset()
    observation = [observation_temp[0],observation_temp[1]]
    time_s = time.time()
    done = False
    for t in range(2000):
    #while not done:
        if i_episode % 100 == 0:
            env.render()
        ##First run through of the network. Take the observation, put it through the network
        action = agent.choose_action(observation)


        observation_temp, reward, done, info = env.step(action)
        observation_ = [observation_temp[0],observation_temp[1]]
        if t==199:
            done = True

        ######RECORDS AND SAVES THE STATE#####
        agent.store_rewards(reward)


        #######################################
        score += reward
        agent.learn(observation, reward, observation_, done)
        #agent.learn()
        observation = observation_
        if done:
           # agent.detach()
            score_history.append(score)
            print("Episode: ", i_episode, "Reward: ", score, "  Time: ", time.time() - time_s)


            break
    if i_episode%100 == 0:
        agent.save_model()
        print("Model Saved")

plotLearning(score_history, filename = "mountaincar_recc_30k.png", window = 10)




env.close()



 tfc4e