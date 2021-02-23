import numpy as np
import random
import torch
import time
import gym
from Agent_Object import PolicyAgent
from Agent_Object import ACAgent
from Agent_Object import ACRecurrentAgent
from Regular_Network import RecurrentNetwork
from pole_environment import CartPoleEnv
from utils import plotLearning
env = CartPoleEnv()
# Variable Definitions
isize = 1 #env.observation_space.shape[0]
seed = 2
hsize = [128,128]
num_actions = env.action_space.n
model_save_name = "pole_balance_1obs_recc"
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
agent2 = ACRecurrentAgent(device, [0.001, 0.005], isize, gamma, num_actions, hsize, 2,2, model_save_name)


#the current agent
agent = agent2


score_history = []
score = 0
max_episodes = 40000
time_s = time.time()

predict_temp = RecurrentNetwork(0.0001, 1, hsize, 1, device, False, 1, 1)



for i_episode in range(max_episodes):
    score = 0
    observation_temp = env.reset()
    observation = [observation_temp[2]]
    agent.zero_loss()

    for t in range(1000):
        predict_temp.optimizer.zero_grad()

        action = agent.choose_action(observation)


        observation_temp, reward, done, info = env.step(action)
        #[predicted_v, rec_v] = predict_temp.forward([observation_temp[2]], rec_v)
        #loss = (predicted_v-actual_v) ** 2
        #loss_tot.append(loss.cpu().data.numpy())
        #loss.backward(retain_graph=True)

        #predict_temp.optimizer.step()
        #actual_v = observation_temp[3]

        observation_ = [observation_temp[0]]#predicted_v]


        ######RECORDS AND SAVES THE STATE#####
        agent.store_rewards(reward)
        agent.add_loss(observation, reward, observation_, done)


        #######################################
        score += reward

        observation = observation_
        if done:
            agent.learn()
            #if i_episode%100 == 0:
            print("Episode: ", i_episode, "Reward: ", score, "  Time: ", time.time() - time_s) #"Loss AVG:", np.mean(loss_tot), "Obs at end. Actual:", actual_v, "Pred:", predicted_v.cpu().data)
            time_s = time.time()
            agent.clear()
            score_history.append(score)



            break
    if i_episode%100 == 0:

        agent.save_model()
        #print("Model Saved")

plotLearning(score_history, filename = "pole_balance_recc_test.png", window = 10)
env.close()



