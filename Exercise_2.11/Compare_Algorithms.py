
'''
Goal: Compare the average reward over the last 1000 steps for each of the following algorithms in the below context:

Context:
Non stationary case
200,000 steps

Record and plot average reward over last 100,000 steps

Algorithms: 
1) constant step-size e-greedy algorithm ; alpha = 0.1
2) greedy with optimisitc initialization ; alpha = 0.1
3) UCB
4) Gradient bandit

'''

'''
TODO

Environment for test

Start simple and just try to get e-greedy working for test - recording last x steps
Implement greedy with optimistic initialization
Implement UCB
Implement Gradient Bandit

'''

from enum import IntEnum

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

np.random.seed(42)

class Methods(IntEnum):
    e_greedy = 0
    optimistic = 1      # Greedy with optimistic initialization
    ucb = 2
    gradient = 3


# Experiment Variables
Average_Reward = np.zeros(shape=(4), dtype=float)

NUM_RUNS = 2000     # Ideally want 1000 or 2000 like what Exercise 2.5 wanted
NUM_STEPS = 1000    # want 200_000

NUM_FIRST_STEPS = 1000

# e-greedy variables
k = 10
epsilon = 0.1
alpha = 0.1

# Using incremental method to update Q, N, R so only need to keep one value per action
q = np.zeros( shape=(k), dtype=float)
Q = np.zeros( shape=(len(Methods) - 1, k), dtype=float)
N = np.full( shape=(len(Methods) - 1, k), fill_value=1e-9,dtype=float)    # initialize with non zero so we can compute UCB

H = np.zeros( shape=(k), dtype=float)
probabilities = np.empty(shape=H.shape)

R_average = np.zeros( shape=(len(Methods), 10), dtype=float)

def run_experiment():

    methods = ['e-greedy', 'optimistic', 'ucb', 'gradient'] 

    for method in methods: 

        if method == 'e-greedy':
            params =  np.geomspace(1/128, 1/4, 6)

        elif method == 'optimistic':
            params = np.geomspace(1/4, 4, 5)

        elif method == 'ucb':
            params = np.geomspace(1/16, 4, 7)

        elif method == 'gradient':
            params = np.geomspace(1/32, 3, 7)   # this one is a bit weird because it ends at 3 or something close to it


        # TODO : Need to figure out loop for params and make them fit for each method - ask chat how to do
        for param_index, param_value in tqdm(enumerate(params)):

            # Param Q initialization for greedy optimistic
            Q[Methods.optimistic] = np.full(shape=(k), fill_value=param_value)

            R_total_all_runs_avg = np.zeros( shape=(len(Methods)) ,dtype=float)

            for run in tqdm(range(NUM_RUNS)):
                
                # TODO: Have other initializations
                reset_bandit_problem(param_value)    # TODO - make sure this doesn't mess with any others bc want this to be indp

                R_total_reward_single_run = np.zeros( shape=(len(Methods)) ,dtype=float)

                for time in range(NUM_STEPS):

                    randomWalk(q)

                    if method == 'e-greedy':
                        action = get_e_greedy_action(epsilon=param_value, Q=Q[Methods.e_greedy])
                        reward = get_reward(action)
                        Q[Methods.e_greedy][action] = Q[Methods.e_greedy][action] + ( alpha * (reward - Q[Methods.e_greedy][action]))
                        N[Methods.e_greedy][action] += 1

                        if time < NUM_FIRST_STEPS:
                            R_total_reward_single_run[Methods.e_greedy] += reward


                    # Greedy with optimistic initialization, alpha = 0.1
                    if method == 'optimistic':
                        action = get_e_greedy_action(epsilon=0, Q=Q[Methods.optimistic])
                        reward = get_reward(action)
                        Q[Methods.optimistic][action] = Q[Methods.optimistic][action] + ( alpha * (reward - Q[Methods.optimistic][action]))
                        N[Methods.optimistic][action] += 1
                        if time < NUM_FIRST_STEPS:
                            R_total_reward_single_run[Methods.optimistic] += reward
                    
                    if method == 'ucb':
                        action = get_ucb_action(c=param_value, t = time, Q=Q[Methods.ucb], N=N[Methods.ucb])
                        reward = get_reward(action)
                        Q[Methods.ucb][action] = Q[Methods.ucb][action] + ( alpha * (reward - Q[Methods.ucb][action]))
                        N[Methods.ucb][action] += 1
                        if time < NUM_FIRST_STEPS:
                            R_total_reward_single_run[Methods.ucb] += reward

                    if method == 'gradient':
                        # picks action from softmax prob distribution
                        # updates preferences using stochastic gradient ascent
                        # does not use Q or N values but just trial and error with preference
                        action = get_gradient_action(H)
                        reward = get_reward(action)

                        if time < NUM_FIRST_STEPS:
                            R_total_reward_single_run[Methods.gradient] += reward

                        baseline = R_total_reward_single_run[Methods.gradient] / (time + 1)
                        H[action] = H[action] + (param_value*(reward - baseline)*(1 - probabilities[action]))

                        for a in range(k):
                            if a == action:
                                continue
                            
                            H[a] = H[a] - (param_value*(reward - baseline)*probabilities[a])


                if method == 'e-greedy':
                    R_total_all_runs_avg[Methods.e_greedy] += (R_total_reward_single_run[Methods.e_greedy] / NUM_STEPS)
                
                if method == 'optimistic':
                    R_total_all_runs_avg[Methods.optimistic] += (R_total_reward_single_run[Methods.optimistic] / NUM_STEPS)

                if method == 'ucb':
                    R_total_all_runs_avg[Methods.ucb] += (R_total_reward_single_run[Methods.ucb] / NUM_STEPS)

                if method == 'gradient':
                    R_total_all_runs_avg[Methods.gradient] += (R_total_reward_single_run[Methods.gradient] / NUM_STEPS)

            if method == 'e-greedy':
                R_average[Methods.e_greedy][param_index] = (R_total_all_runs_avg[Methods.e_greedy] / NUM_FIRST_STEPS)
            
            if method == 'optimistic':
                R_average[Methods.optimistic][param_index] = (R_total_all_runs_avg[Methods.optimistic] / NUM_FIRST_STEPS)

            if method == 'ucb':
                R_average[Methods.ucb][param_index] = (R_total_all_runs_avg[Methods.ucb] / NUM_FIRST_STEPS)

            if method == 'gradient':
                R_average[Methods.gradient][param_index] = (R_total_all_runs_avg[Methods.gradient] / NUM_FIRST_STEPS)



    # Define x-values (log-spaced)
    x = np.geomspace(1/128, 4, 10)

    # Create labels as fractions
    tick_labels = ["1/128", "1/64", "1/32", "1/16", "1/8", "1/4", "1/2", "1", "2", "4"]

    fig, ax = plt.subplots()
    ax.set_xscale('log')

    x_greedy_param = np.geomspace(1/128, 1/4, 6)
    x_optimistic_param = np.geomspace(1/4, 4, 5)
    x_ucb_param = np.geomspace(1/16, 4, 7)
    x_gradient_param = np.geomspace(1/32, 3, 7)

    print("R_avg values: ")
    print(R_average[Methods.e_greedy])
    print(R_average[Methods.optimistic])
    print(R_average[Methods.ucb])
    print(R_average[Methods.gradient])

    print("Trimmed: R_avg values: ")
    print(np.trim_zeros(R_average[Methods.e_greedy], 'b'))
    print(np.trim_zeros(R_average[Methods.optimistic], 'b'))
    print(np.trim_zeros(R_average[Methods.ucb], 'b'))
    print(np.trim_zeros(R_average[Methods.gradient], 'b'))

    ax.plot(x_greedy_param, np.trim_zeros(R_average[Methods.e_greedy], 'b'), label='e_greedy', color='red')
    ax.plot(x_optimistic_param, np.trim_zeros(R_average[Methods.optimistic], 'b'), label='optimistic', color='black')
    ax.plot(x_ucb_param, np.trim_zeros(R_average[Methods.ucb], 'b'), label='ucb', color='blue')
    ax.plot(x_gradient_param, np.trim_zeros(R_average[Methods.gradient], 'b'), label='gradient', color='green')

    # Set custom ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)

    # Add axis labels and title
    ax.set_xlabel(r"$\varepsilon \quad \alpha \quad c \quad Q_0$", fontsize=14)
    ax.set_ylabel("Average reward\nover first 1000 steps", fontsize=14)
    ax.set_title("Parameter Study")

    # Optional styling
    ax.tick_params(axis='x', rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.show()



# Helper functions
def get_gradient_action(H):
    global probabilities

    softmax_denominator = 0
    for h in H:
        softmax_denominator += np.exp(h)

    probabilities = np.empty(shape=H.shape)
    
    for i, h in enumerate(H):
        probabilities[i] = (np.exp(h) / softmax_denominator)
    
    # Return index of random choice with probabilities generated from softmax distribution
    return np.random.choice(np.arange(k), p=probabilities)

def get_ucb_action(c, t, Q, N):

    UCB = c * np.sqrt(np.log(t)) * np.sqrt((N**-1))
    Q_ucb = Q + UCB
    return np.argmax(Q_ucb)

def reset_bandit_problem(param_value):

    global q 
    global Q
    global N 

    q = np.zeros( shape=(k), dtype=float)
    Q = np.zeros( shape=(len(Methods), k), dtype=float)
    N = np.full( shape=(len(Methods) - 1, k), fill_value=1e-9,dtype=float)    # initialize with non zero so we can compute UCB

    Q[Methods.optimistic] = np.full(shape=(k), fill_value=param_value)

def randomWalk(x):

    walk_set = [-0.1, 0, 0.1]
    for i in range(len(x)):
        x[i] += np.random.choice(walk_set)

def get_e_greedy_action(epsilon, Q):

    if epsilon == 0:
        doExplore = 0
    else:
        doExplore = np.random.binomial(1, epsilon)

    if doExplore:
        action = np.random.randint(k)

    else:
        # action = Q.index(max(Q))
        action = np.argmax(Q)

    return action
    
def get_reward(action_index):
    reward_mean = q[action_index]
    variance = 1
    stdv = np.sqrt(variance)
    return np.random.normal(reward_mean, stdv)

if __name__ == '__main__':
    run_experiment()        

