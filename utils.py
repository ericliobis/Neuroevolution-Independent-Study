#import matplotlib.pyplot as plt

#import tensorflow as tf
import numpy as np
import gym

def discount_and_normalize_rewards(ep_rewards, gamma):
    discounted_episode_rewards = np.zeros_like(ep_rewards)
    cumulative = 0.0
    for i in reversed(range(len(ep_rewards))):
        cumulative = cumulative * gamma + ep_rewards[i]
        discounted_episode_rewards[i] = cumulative

    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards-mean)/(std)
    return discounted_episode_rewards



def plotLearning(scores, filename, x=None, window=5):
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Score')
    plt.xlabel('Game')
    plt.plot(x, running_avg)
    plt.savefig(filename)