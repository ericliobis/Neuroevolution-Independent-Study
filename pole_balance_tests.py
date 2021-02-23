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
#import matplotlib.pyplot as plt
#import matplotlib.colors as colors
#from matplotlib import cm
#from numpy import genfromtxt


config = configparser.ConfigParser()
config.read('config.ini')
seed =  int(config['DEFAULT']['seed'])
act_fun = [int(x) for x in config.get('NETWORK_ACT', 'act_fun').split(',')]
hsize = [int(x) for x in config.get('NETWORK_SHAPE', 'hsize').split(',')]
##Create the environment
env = CartPoleEnv_Rand_Length(seed)


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





def runNetworks(learn, network_type, network_save_name, save_eps,  max_ep, episode_length, render, render_eps, target_range_lower, target_range_upper):
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

    dist_history_all = []
    balance_history_all = []
    score_history_all = []





    score_history = []
    max_episodes = max_ep
    ep_len = episode_length
    save = []

    #Create the right network depending on the type
    if network_type == 0:
        agent = PolicyAgent([learning_rate], isize, hsize, num_actions, act_fun, device, gamma, eps, 0, batch_size, save_path,network_save_name)
        save.append(save_path + "Regular_agent_" + network_save_name);
    else:
        save.append(save_path + "policy_" + network_save_name)
        save.append(save_path + "critic_" + network_save_name)
        if network_type == 1:
            agent = ACAgent([learning_rate_policy, learning_rate_critic], isize, hsize, num_actions,  act_fun, device, gamma, eps, weight_decay, blossv, batch_size, save_path,network_save_name)
        elif network_type == 2:
            agent = ACRecurrentAgent([learning_rate_policy, learning_rate_critic], isize, hsize, num_actions,  act_fun, rec_layer_out, rec_layer_in,device, gamma, eps, weight_decay, blossv, batch_size, save_path,network_save_name)
        elif network_type == 3:
            agent = ACHebbRecurrentAgent([learning_rate_policy, learning_rate_critic], isize, hsize, num_actions,  act_fun, rec_layer_out, rec_layer_in,device, gamma, eps, weight_decay, blossv, batch_size, save_path,network_save_name)


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
        dist_history_all.append(dist_hist);
        balance_history_all.append(balance_hist);
        score_history_all.append(score_hist);
        if i_episode%save_eps == 0:
            agent.save_model()
            print("Model Saved")

    #plotLearning(score_history, filename = "cartpole.png", window = 10)
    env.close()
    with open(save_path+"DIST_HIST_"+save_name+ ".csv", "w+", newline = '') as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(dist_history_all)
    with open(save_path+"SCORE_HIST_"+save_name+ ".csv", "w+", newline = '') as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(score_history_all)
    with open(save_path+"BALANCE_HIST_"+save_name+ ".csv", "w+", newline = '') as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(balance_history_all)

def load_and_plot(thing_to_plot):
    plot_name = ""
    if(thing_to_plot ==0):
        plot_name = "DIST_HIST_"
    if (thing_to_plot == 1):
        plot_name = "SCORE_HIST_"
    if (thing_to_plot == 2):
        plot_name = "BALANCE_HIST_"
    test = []
    with open(data_path+plot_name+save_name+ ".csv", "r") as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            test.append(line)

    n = len(test)
    colors = plt.get_cmap("RdYlGn")
    plt.cm.jet
    for i in range(len(test)):

        line = [float(j) for j in test[i]]
        print(line)
        plt.plot(line)
        plt.xlabel('Time')
        plt.ylabel('test')
    plt.show()


def ACHebbAgent_test(learn, record):
    score_history = []
    continue_history = []
    start = -3
    stop = 3.5
    incr = 0.5
    ep_len = 1000

    starts = 100
    continues = 1



    range1 = np.arange(start, stop, incr)
    range1 = [-0.5];
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
                    if i_episode % 1000 == 0:
                        env.render()
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









def PolicyAgent_test(learn):
    score_history = np.array([])
    continue_history = []
    start = -3
    stop = 3.5
    incr = 0.5
    ep_len = 1000

    starts = 10
    continues = 10


    range1 = [-3];
    #range1 = np.arange(start, stop, incr)
    print("Range: ", range1)


    for i_targ in np.nditer(range1):
        print("Target: ", i_targ)

        env.setTarget(i_targ)
        continue_history = np.zeros(continues)

        for i_starts in range(starts):
            #reload the model at each start

            if (path.exists(agent1_name)):
                agent1.load_model(agent1_name)

            for i_episode in range(continues):

                #reset the environment
                score = 0
                observation_temp = env.reset()
                observation = observation_temp;

                for t in range(ep_len):
                    if i_episode % 1 == 0 and i_starts == 0:
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
                    if done  or t == ep_len-1:
                        #score_history.append(score)
                        if (learn):
                            agent1.learn()


                        print("==========================================")
                        print("Episode: ", i_episode)
                        print("Reward: ", score)
                        print("Length:", env.length)
                        print("Target: ", i_targ)

                        break
                continue_history[i_episode] += score

        score_history = np.append(score_history,continue_history/starts)

    np.savetxt("PG1.csv", score_history, delimiter=",")




    env.close()


def train_maze(paramdict):
    # params = dict(click.get_current_context().params)
    TOTALNBINPUTS = 9+4+4;
    RFSIZE = 3
    ADDITIONALINPUTS = 4
    # TOTALNBINPUTS =  RFSIZE * RFSIZE + ADDITIONALINPUTS + NBNONRESTACTIONS
    print("Starting training...")
    params = {}
    # params.update(defaultParams)
    params.update(paramdict)
    print("Passed params: ", params)
    print(platform.uname())
    # params['nbsteps'] = params['nbshots'] * ((params['prestime'] + params['interpresdelay']) * params['nbclasses']) + params['prestimetest']  # Total number of steps per episode
    suffix = "btchFixmod_" + "".join([str(x) + "_" if pair[0] is not 'nbsteps' and pair[0] is not 'rngseed' and pair[
        0] is not 'save_every' and pair[0] is not 'test_every' and pair[0] is not 'pe' else '' for pair in
                                      sorted(zip(params.keys(), params.values()), key=lambda x: x[0]) for x in pair])[
                             :-1] + "_rngseed_" + str(
        params['rngseed'])  # Turning the parameters into a nice suffix for filenames

    # Initialize random seeds (first two redundant?)
    print("Setting random seeds")
    np.random.seed(params['rngseed']);
    random.seed(params['rngseed']);
    torch.manual_seed(params['rngseed'])

    print("Initializing network")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    net = ACHebbRecurrentAgent([params['lr'], params['lr']], 9+4+4, [params['hs']], 4,  [1,0],0, 0,device, params['gr'], 1e-4, params['l2'], params['blossv'], params['bs'], save_path,"test")

    #print("Shape of all optimized parameters:", [x.size() for x in net.parameters()])
    #allsizes = [torch.numel(x.data.cpu()) for x in net.parameters()]
    #print("Size (numel) of all optimized elements:", allsizes)
    #print("Total size (numel) of all optimized elements:", sum(allsizes))

    # total_loss = 0.0
    print("Initializing optimizer")

    # optimizer = torch.optim.SGD(net.parameters(), lr=1.0*params['lr'])
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=params['gamma'], step_size=params['steplr'])

    BATCHSIZE = params['bs']

    LABSIZE = params['msize']
    lab = np.ones((LABSIZE, LABSIZE))
    CTR = LABSIZE // 2

    # Grid maze
    lab[1:LABSIZE - 1, 1:LABSIZE - 1].fill(0)
    for row in range(1, LABSIZE - 1):
        for col in range(1, LABSIZE - 1):
            if row % 2 == 0 and col % 2 == 0:
                lab[row, col] = 1
    # Not strictly necessary, but cleaner since we start the agent at the
    # center for each episode; may help loclization in some maze sizes
    # (including 13 and 9, but not 11) by introducing a detectable irregularity
    # in the center:
    lab[CTR, CTR] = 0

    all_losses = []
    all_grad_norms = []
    all_losses_objective = []
    all_total_rewards = []
    all_losses_v = []
    lossbetweensaves = 0
    nowtime = time.time()
    meanrewards = np.zeros((LABSIZE, LABSIZE))
    meanrewardstmp = np.zeros((LABSIZE, LABSIZE, params['eplen']))

    pos = 0

    # pw = net.initialZeroPlasticWeights()  # For eligibility traces

    # celoss = torch.nn.CrossEntropyLoss() # For supervised learning - not used here

    print("Starting episodes!")

    for numiter in range(params['nbiter']):
        net.zero_loss()
        PRINTTRACE = 0
        if (numiter + 1) % (params['pe']) == 0:
            PRINTTRACE = 1

        # lab = makemaze.genmaze(size=LABSIZE, nblines=4)
        # count = np.zeros((LABSIZE, LABSIZE))

        # Select the reward location for this episode - not on a wall!
        # And not on the center either! (though not sure how useful that restriction is...)
        # We always start the episode from the center
        posr = {};
        posc = {}
        rposr = {};
        rposc = {}
        for nb in range(BATCHSIZE):
            # Note: it doesn't matter if the reward is on the center (see below). All we need is not to put it on a wall or pillar (lab=1)
            myrposr = 0;
            myrposc = 0
            while lab[myrposr, myrposc] == 1 or (myrposr == CTR and myrposc == CTR):
                myrposr = np.random.randint(1, LABSIZE - 1)
                myrposc = np.random.randint(1, LABSIZE - 1)
            rposr[nb] = myrposr;
            rposc[nb] = myrposc
            # print("Reward pos:", rposr, rposc)
            # Agent always starts an episode from the center
            posc[nb] = CTR
            posr[nb] = CTR


        reward = np.zeros(BATCHSIZE)
        sumreward = np.zeros(BATCHSIZE)
        rewards = []
        vs = []
        logprobs = []
        dist = 0
        numactionschosen = np.zeros(BATCHSIZE, dtype='int32')

        # reloctime = np.random.randint(params['eplen'] // 4, (3 * params['eplen']) // 4)

        # print("EPISODE ", numiter)
        for numstep in range(params['eplen']):

            inputs = np.zeros((BATCHSIZE, TOTALNBINPUTS), dtype='float32')

            labg = lab.copy()
            for nb in range(BATCHSIZE):
                inputs[nb, 0:RFSIZE * RFSIZE] = labg[posr[nb] - RFSIZE // 2:posr[nb] + RFSIZE // 2 + 1,
                                                posc[nb] - RFSIZE // 2:posc[nb] + RFSIZE // 2 + 1].flatten() * 1.0

                # Previous chosen action
                inputs[nb, RFSIZE * RFSIZE + 1] = 1.0  # Bias neuron
                inputs[nb, RFSIZE * RFSIZE + 2] = numstep / params['eplen']
                inputs[nb, RFSIZE * RFSIZE + 3] = 1.0 * reward[nb]
                inputs[nb, RFSIZE * RFSIZE + ADDITIONALINPUTS + numactionschosen[nb]] = 1

            #inputsC = torch.from_numpy(inputs).to(device)


            actionschosen = net.choose_action(inputs)

            numactionschosen = actionschosen  # We want to break gradients
            reward = np.zeros(BATCHSIZE, dtype='float32')

            for nb in range(BATCHSIZE):
                myreward = 0
                numactionchosen = numactionschosen[nb]

                tgtposc = posc[nb]
                tgtposr = posr[nb]
                if numactionchosen == 0:  # Up
                    tgtposr -= 1
                elif numactionchosen == 1:  # Down
                    tgtposr += 1
                elif numactionchosen == 2:  # Left
                    tgtposc -= 1
                elif numactionchosen == 3:  # Right
                    tgtposc += 1
                else:
                    raise ValueError("Wrong Action")

                reward[nb] = 0.0  # The reward for this step
                if lab[tgtposr][tgtposc] == 1:
                    reward[nb] -= params['wp']
                else:
                    posc[nb] = tgtposc
                    posr[nb] = tgtposr

                # Did we hit the reward location ? Increase reward and teleport!
                # Note that it doesn't matter if we teleport onto the reward, since reward hitting is only evaluated after the (obligatory) move...
                # But we still avoid it.
                if rposr[nb] == posr[nb] and rposc[nb] == posc[nb]:
                    reward[nb] += params['rew']
                    posr[nb] = np.random.randint(1, LABSIZE - 1)
                    posc[nb] = np.random.randint(1, LABSIZE - 1)
                    while lab[posr[nb], posc[nb]] == 1 or (rposr[nb] == posr[nb] and rposc[nb] == posc[nb]):
                        posr[nb] = np.random.randint(1, LABSIZE - 1)
                        posc[nb] = np.random.randint(1, LABSIZE - 1)

            rewards.append(reward)
            net.store_rewards(np.array(reward))
            sumreward += reward

            # This is an "entropy penalty", implemented by the sum-of-squares of the probabilities because our version of PyTorch did not have an entropy() function.
            # The result is the same: to penalize concentration, i.e. encourage diversity in chosen actions.

            # if PRINTTRACE:
            #    print("Step ", numstep, " Inputs (to 1st in batch): ", inputs[0, :TOTALNBINPUTS], " - Outputs(1st in batch): ", y[0].data.cpu().numpy(), " - action chosen(1st in batch): ", numactionschosen[0],
            #            #" - mean abs pw: ", np.mean(np.abs(pw.data.cpu().numpy())),
            #            " -Reward (this step, 1st in batch): ", reward[0])

        # Epi
        all_total_rewards.append(sumreward.mean())
        net.learn()
        if (numiter + 1) % params['pe'] == 0:
            print(numiter, "====")
            print("Mean loss: ", lossbetweensaves / params['pe'])
            lossbetweensaves = 0
            print("Mean reward (across batch and last", params['pe'], "eps.): ",
                  np.sum(all_total_rewards[-params['pe']:]) / params['pe'])
            # print("Mean reward (across batch): ", sumreward.mean())
            previoustime = nowtime
            nowtime = time.time()
            print(sumreward)
            print("Time spent on last", params['pe'], "iters: ", nowtime - previoustime)
            # print("ETA: ", net.eta.data.cpu().numpy(), " etaet: ", net.etaet.data.cpu().numpy())



def train_maze_1():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rngseed", type=int, help="random seed", default=0)
    parser.add_argument("--rew", type=float,
                        help="reward value (reward increment for taking correct action after correct stimulus)",
                        default=10.0)
    parser.add_argument("--wp", type=float, help="penalty for hitting walls", default=.0)
    parser.add_argument("--bent", type=float,
                        help="coefficient for the entropy reward (really Simpson index concentration measure)",
                        default=0.03)
    parser.add_argument("--blossv", type=float, help="coefficient for value prediction loss", default=.1)
    parser.add_argument("--msize", type=int, help="size of the maze; must be odd", default=11)
    parser.add_argument("--gr", type=float, help="gammaR: discounting factor for rewards", default=.9)
    parser.add_argument("--gc", type=float, help="gradient norm clipping", default=4.0)
    parser.add_argument("--lr", type=float, help="learning rate (Adam optimizer)", default=1e-4)
    parser.add_argument("--eplen", type=int, help="length of episodes", default=200)
    parser.add_argument("--hs", type=int, help="size of the recurrent (hidden) layer", default=100)
    parser.add_argument("--bs", type=int, help="batch size", default=30)
    parser.add_argument("--l2", type=float, help="coefficient of L2 norm (weight decay)", default=0)  # 3e-6
    parser.add_argument("--nbiter", type=int, help="number of learning cycles", default=200000)
    parser.add_argument("--save_every", type=int, help="number of cycles between successive save points", default=50)
    parser.add_argument("--pe", type=int, help="number of cycles between successive printing of information",
                        default=10)
    args = parser.parse_args();
    argvars = vars(args);
    argdict = {k: argvars[k] for k in argvars if argvars[k] != None}

    train_maze(argdict)
runNetworks(learn, network, save_name, save_every, max_episodes,length_of_episodes, render, render_eps, target_range_lower, target_range_upper)

#load_and_plot(0)