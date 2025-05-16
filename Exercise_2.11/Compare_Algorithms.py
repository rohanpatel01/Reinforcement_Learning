
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

class Methods(IntEnum):
    e_greedy = 0
    greedy_optimistic_initialization = 1
    ucb = 2
    gradient_bandit = 3


# Experiment Variables
Average_Reward = np.zeros(shape=(4), dtype=float)

NUM_RUNS = 10      # Ideally want 1000 or 2000 like what Exercise 2.5 wanted
NUM_STEPS = 1000    # want 200_000

LAST_STEPS_FRACTION = 1 - (0.5)       # Record this last fraction of timesteps
NUM_LAST_STEPS = NUM_RUNS * LAST_STEPS_FRACTION    # start recording rewards after this many time steps have passed
START_OF_LAST = NUM_STEPS - NUM_LAST_STEPS


start = 1/128
end = 1/4
num = 6             # Full experiment has 10

# params = np.empty(shape=(len(Methods)))
# params[Methods.e_greedy] = np.geomspace(start, end, num)

params = np.geomspace(start, end, num)

# TODO ^^^^ Note: Need to configure param range for each method
# ex: e-greedy param cannot go above 1
# look at graph and do same for others 

RANDOM_WALK_MEAN = 0
RANDOM_WALK_STANDARD_DEVIATION = 0.01


# e-greedy variables
k = 10
epsilon = 0.1
alpha = 0.1

# Using incremental method to update Q, N, R so only need to keep one value per action
q = np.zeros( shape=(k), dtype=float)
Q = np.zeros( shape=(len(Methods), k), dtype=float)
N = np.zeros( shape=(len(Methods), k), dtype=int)
R_average = np.zeros( shape=(len(Methods), len(params)), dtype=float)


def run_experiment():

    # TODO : Need to figure out loop for params and make them fit for each method - ask chat how to do
    for param_index, param_value in tqdm(enumerate(params)):
        for run in tqdm(range(NUM_RUNS)):

            # TODO: Have other initializations
            # reset_bandit_problem()    # TODO - make sure this doesn't mess with any others bc want this to be indp

            for time in range(NUM_STEPS):

                randomWalk(q, RANDOM_WALK_MEAN, RANDOM_WALK_STANDARD_DEVIATION)

                # Method 1: E-greedy 
                e_greedy_step(time, param_index)

        
        R_average[Methods.e_greedy][param_index] /= NUM_LAST_STEPS
       

    # Printing
    print(NUM_LAST_STEPS)
    print(START_OF_LAST)
    print("q: ", q)
    print("R avg: ", R_average)

    # Plot results
    x = params 
    y = R_average[Methods.e_greedy]
    figure, axes = plt.subplots()
    axes.plot(x, y)
    # axes.title("Parameter Study")
    plt.show()



# Helper functions
def e_greedy_step(time, param_index):
    global Q
    global N
    global R_average

    action = get_e_greedy_action(epsilon, Q[Methods.e_greedy])
    reward = get_reward(action)
    Q[Methods.e_greedy][action] = Q[Methods.e_greedy][action] + ( alpha * (reward - Q[Methods.e_greedy][action]))
    N[Methods.e_greedy][action] += 1

    if time >= START_OF_LAST:
        R_average[Methods.e_greedy][param_index] += reward

def reset_bandit_problem():

    global q 
    global Q
    global N 

    # True q*(a) values (Will use this to compute the rewards)
    q = np.array([0 for _ in range(k)]) 

    # Running value for incremental value for all actions
    Q = np.array([None] * 2)
    Q[0] = np.array([0 for _ in range(k)]) # Estimated Q for method 1
    Q[1] = np.array([0 for _ in range(k)]) # Estimated Q for method 2

    # Number of times we've seen action thus far
    N = np.array([None] * 2)
    N[0] = np.array([0 for _ in range(k)])
    N[1] = np.array([0 for _ in range(k)])

def randomWalk(x, mean, standard_deviation):

    walk_set = [-0.1, 0, 0.1]
    for i in range(len(x)):
        x[i] += np.random.choice(walk_set)

def get_e_greedy_action(epsilon, Q):

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

