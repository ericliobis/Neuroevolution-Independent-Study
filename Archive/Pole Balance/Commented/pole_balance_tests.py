import numpy as np
import random
import torch
import gym
from os import path
from Agents import PolicyAgent
from Agents import ACAgent
from Agents import ACRecurrentAgent
from Agents import ACHebbRecurrentAgent
#from pole_environment import CartPoleEnv
from pole_environment import CartPoleEnv_Rand_Length
from utils import plotLearning
env = CartPoleEnv_Rand_Length()
# Variable Definitions
isize =env.observation_space.shape[0]
seed = 1
hsize = [4]
act_fun = [3, 0]
num_actions = env.action_space.n
print("act:", num_actions)
model_save_name = "pole_balance_1obs"
learning_rate = 0.01
learning_rate_policy = 0.001
learning_rate_critic = 0.01
rec_layer_in = 0
rec_layer_out = 0
batch_size = 1
gamma = 0.99
blossv = 0.1
eps = 1e-4
weight_decay=0.1


#sets the seed
np.random.seed(seed);
random.seed(seed);
torch.manual_seed(seed)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

a1 = "Policy_Model_small_target"
a2 = "AC_model_small_target"
a3 = "AC_rec_model_small_target"
a4 = "AC_heb_model_small_target"

agent1_name = "Regular_agent_" + a1
agent2_name_policy = "policy_" + a2
agent3_name_policy = "policy_" + a3
agent4_name_policy = "policy_" + a4
agent2_name_critic = "critic_" + a2
agent3_name_critic = "critic_" + a3
agent4_name_critic = "critic_" + a4


agent1 = PolicyAgent([learning_rate], isize, hsize, num_actions,  act_fun,device, gamma, eps, 0, batch_size, a1)
agent2 = ACAgent([learning_rate_policy, learning_rate_critic], isize, hsize, num_actions,  act_fun, device, gamma, eps, weight_decay, blossv, batch_size, a2)
agent3 = ACRecurrentAgent([learning_rate_policy, learning_rate_critic], isize, hsize, num_actions,  act_fun, rec_layer_out, rec_layer_in,device, gamma, eps, weight_decay, blossv, batch_size, a3)
agent4 = ACHebbRecurrentAgent([learning_rate_policy, learning_rate_critic], isize, hsize, num_actions,  act_fun, rec_layer_out, rec_layer_in,device, gamma, eps, weight_decay, blossv, batch_size, a4)


def PolicyAgent():

    score_history = []
    score = 0
    max_episodes = 50000
    if(path.exists(agent1_name)):
        agent1.load_model(agent1_name)




    for i_episode in range(max_episodes):
        #env.setLen()
        env.setTarget(random.uniform(-3,3))
        score = 0
        observation_temp = env.reset()
        observation = observation_temp;

        for t in range(1000):
            if i_episode % 10 == 0:
                env.render()
            ##First run through of the network. Take the observation, put it through the network
            action = agent1.choose_action(np.asarray([observation], dtype=np.float32))


            observation_temp, reward, done, info = env.step(action)
            observation_ = observation_temp



            ######RECORDS AND SAVES THE STATE#####
            agent1.store_rewards(reward)


            #######################################
            score += reward

            observation = observation_
            if done or t == 999:
                score_history.append(score)
                agent1.learn()


                print("==========================================")
                print("Episode: ", i_episode)
                print("Reward: ", score)
                print("Length:", env.length)

                break
        if i_episode%100 == 0:
            agent1.save_model()
            print("Model Saved")

    plotLearning(score_history, filename = "cartpole.png", window = 10)
    env.close()

def ACAgent():

    score_history = []
    score = 0
    max_episodes = 50000
    if (path.exists(agent2_name_policy) and  path.exists(agent2_name_critic)):
        agent2.load_model(agent2_name_policy,agent2_name_critic)



    for i_episode in range(max_episodes):
        #env.setLen()
        agent2.zero_loss()
        score = 0
        observation_temp = env.reset()
        observation = observation_temp

        for t in range(1000):
            if i_episode % 100 == 0:
                env.render()
            ##First run through of the network. Take the observation, put it through the network
            action = agent2.choose_action(np.asarray([observation], dtype=np.float32))


            observation_temp, reward, done, info = env.step(action[0])
            observation_ = observation_temp



            ######RECORDS AND SAVES THE STATE#####
            agent2.store_rewards(np.array(reward))


            #######################################
            score += reward

            observation = observation_
            if done or t == 999:
                score_history.append(score)
                agent2.learn()


                print("==========================================")
                print("Episode: ", i_episode)
                print("Reward: ", score)
                print("Length:", env.length)

                break
        if i_episode%100 == 0:
            agent2.save_model()
            print("Model Saved")

    plotLearning(score_history, filename = "cartpole.png", window = 10)
    env.close()

def ACRecAgent():

    score_history = []
    score = 0
    max_episodes = 50000
    if (path.exists(agent3_name_policy) and  path.exists(agent3_name_critic)):
        agent3.load_model(agent3_name_policy,agent3_name_critic)

    for i_episode in range(max_episodes):
        #env.setLen()
        agent3.zero_loss()
        score = 0
        observation_temp = env.reset()
        observation = observation_temp

        for t in range(1000):
            if i_episode % 100 == 0:
                env.render()
            ##First run through of the network. Take the observation, put it through the network
            action = agent3.choose_action(np.asarray([observation], dtype=np.float32))


            observation_temp, reward, done, info = env.step(action[0])
            observation_ = observation_temp



            ######RECORDS AND SAVES THE STATE#####
            agent3.store_rewards(np.array(reward))


            #######################################
            score += reward

            observation = observation_
            if done:
                score_history.append(score)
                agent3.learn()


                print("==========================================")
                print("Episode: ", i_episode)
                print("Reward: ", score)
                print("Length:", env.length)

                break
        if i_episode%100 == 0:
            agent3.save_model()
            print("Model Saved")

    plotLearning(score_history, filename = "cartpole.png", window = 10)
    env.close()

def ACHebbAgent(learn):

    score_history = []
    score = 0
    max_episodes = 500000
    if (path.exists(agent4_name_policy) and  path.exists(agent4_name_critic)):
        agent4.load_model(agent4_name_policy,agent4_name_critic)

    for i_episode in range(max_episodes):
        #env.setLen()
        env.setTarget(random.uniform(-3, 3))
        agent4.zero_loss()
        score = 0
        observation_temp = env.reset()
        observation = observation_temp

        #observation = [observation_temp[2]]

        for t in range(1000):
            if i_episode % 10 == 0:
                env.render()
            ##First run through of the network. Take the observation, put it through the network
            action = agent4.choose_action(np.asarray([observation], dtype=np.float32))


            observation_temp, reward, done, info = env.step(action[0])
            observation_ = observation_temp



            ######RECORDS AND SAVES THE STATE#####
            agent4.store_rewards(np.array(reward))


            #######################################
            score += reward

            observation = observation_
            if done:
                score_history.append(score)
                if(learn):
                    agent4.learn()


                print("==========================================")
                print("Episode: ", i_episode)
                print("Reward: ", score)
                print("Length:", env.length)

                break
        if i_episode%100 == 0:
            agent4.save_model()
            print("Model Saved")

    plotLearning(score_history, filename = "cartpole.png", window = 10)
    env.close()


def ACHebbAgent_test(learn):
    score_history = []
    score_total = []
    start = 0.1
    stop = 2
    incr = 0.1
    max_episodes = 10
    if (path.exists(agent4_name_policy) and path.exists(agent4_name_critic)):
        agent4.load_model(agent4_name_policy, agent4_name_critic)
    range1 = np.arange(start, stop, incr)
    print("Range: ", range1)


    for i_len in np.nditer(range1):
        env.setLength(i_len)
        total_score = 0
        for i_episode in range(max_episodes):
            agent4.zero_loss()
            score = 0
            observation_temp = env.reset()
            observation = observation_temp

            # observation = [observation_temp[2]]

            for t in range(1000):
                if i_episode % 100 == 0:
                    env.render()
                ##First run through of the network. Take the observation, put it through the network
                action = agent4.choose_action(np.asarray([observation], dtype=np.float32))

                observation_temp, reward, done, info = env.step(action[0])
                observation_ = observation_temp

                ######RECORDS AND SAVES THE STATE#####
                agent4.store_rewards(np.array(reward))

                #######################################
                score += reward

                observation = observation_
                if done:
                    score_history.append(score)
                    if (learn):
                        agent4.learn()

                    print("==========================================")
                    print("Episode: ", i_episode)
                    print("Reward: ", score)
                    print("Length:", env.length)

                    break
            total_score+= score
            if learn and (i_episode % 100 == 0):
                agent4.save_model()
                print("Model Saved")
        total_score = total_score/max_episodes
        score_total.append(total_score)

        #plotLearning(score_history, filename="cartpole.png", window=10)
    env.close()
    print("Range: ", range1)
    print(score_total)


def PolicyAgent_test(learn):
    score_history = []
    score_total = []
    start = 0.1
    stop = 2
    incr = 0.05
    max_episodes = 10
    if(path.exists(agent1_name)):
        agent1.load_model(agent1_name)
    range1 = np.arange(start, stop, incr)
    print("Range: ", range1)


    for i_len in np.nditer(range1):
        env.setLength(i_len)
        total_score = 0
        for i_episode in range(max_episodes):
            score = 0
            observation_temp = env.reset()
            observation = observation_temp;

            for t in range(1000):
                if i_episode % 100 == 0:
                    env.render()
                ##First run through of the network. Take the observation, put it through the network
                action = agent1.choose_action(np.asarray([observation], dtype=np.float32))


                observation_temp, reward, done, info = env.step(action)
                observation_ = observation_temp



                ######RECORDS AND SAVES THE STATE#####
                agent1.store_rewards(reward)


                #######################################
                score += reward

                observation = observation_
                if done:
                    score_history.append(score)
                    if (learn):
                        agent1.learn()


                    print("==========================================")
                    print("Episode: ", i_episode)
                    print("Reward: ", score)
                    print("Length:", env.length)

                    break
            total_score += score
            if learn and (i_episode%100 == 0):
                agent1.save_model()
                print("Model Saved")
        total_score = total_score / max_episodes
        score_total.append(total_score)

    print("Range: ", range1)
    print(score_total)
    env.close()
#PolicyAgent()
#PolicyAgent_test(False)
#ACAgent()
#ACRecAgent()
ACHebbAgent(True)
#ACHebbAgent_test(False)