import numpy as np
import random
import torch
import gym
from Agent_Object import PolicyAgent
from Agent_Object import ACAgent
from pole_environment import CartPoleEnv
from utils import plotLearning
env = CartPoleEnv()
# Variable Definitions
isize = env.observation_space.shape[0]
seed = 1
hsize = [32,32]
num_actions = env.action_space.n

learning_rate = 0.01
gamma = 0.99

print(len(hsize))


#sets the seed
np.random.seed(seed);
random.seed(seed);
torch.manual_seed(seed)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


agent1 = PolicyAgent(device, [learning_rate], isize, gamma, num_actions, hsize)
agent2 = ACAgent(device, [0.0001, 0.0005], isize, gamma, num_actions, hsize)


#the current agent
agent = agent2


score_history = []
score = 0
max_episodes = 3000


for i_episode in range(max_episodes):
    score = 0
    observation = env.reset()



    for t in range(1000):
        #env.render()
        ##First run through of the network. Take the observation, put it through the network
        action = agent.choose_action(observation)


        observation_, reward, done, info = env.step(action)


        ######RECORDS AND SAVES THE STATE#####
        agent.store_rewards(reward)


        #######################################
        score += reward
        agent.learn(observation, reward, observation_, done)
        observation = observation_
        if done:
            score_history.append(score)


            print("==========================================")
            print("Episode: ", i_episode)
            print("Reward: ", score)

            break
    if i_episode%100 == 0:
        agent.save_model()
        print("Model Saved")

plotLearning(score_history, filename = "cartpole.png", window = 10)
env.close()



