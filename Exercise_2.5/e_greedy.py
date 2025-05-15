
# Exercise Description
'''
Goal: Demonstrate difficulties that sample-average method has for non-stationary problems

Modified version of 10-armed testbed
    - all true q*(a) start out equal
    - take normally distr (mean 0, stdv 0.01) indp random walks on each step - each action gets same indp walk amount added
    - prepare plot for action-value method using sample averages, incrementally computed
    - anohter action-value method using a constant step0size parameter, a = 0.1
    - Use e = 0.1
    - Use 10,000 steps


    We are comparing two methods: 
    both with e = 0.1 (e-greedy with uniform probability over ALL actions) and 10,000 steps

    - Method 1: Case that does not work well in non-stationary problems: 
        action-value method using sample averages, incrementall computed

    - Method 2: Case that does work in non-stationary case (with constant step size param)
        action-value method using constant step-size parameter, a = 0.1


    For now just do one run, later can do many runs then average to get average reward and % optimal action

'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm


# SEED = 42
RANDOM_WALK_MEAN = 0
RANDOM_WALK_STANDARD_DEVIATION = 0.01
k = 10  # 10 Actions are: 0 1 2 3 4 5 6 7 8 9
epsilon = 0.1
alpha = 0.1     # constant step size parameter for method 2
TIME_STEPS = 10_000
# np.random.seed(SEED)

q = None
Q = None
N = None
# R = None
R_average = None
num_optimal_actions = None

R_average = np.zeros(shape=(2, TIME_STEPS), dtype=np.float64)
num_optimal_actions = np.zeros((2, TIME_STEPS), dtype=np.float64)

def initialize_bandit_problem():

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

'''
Take e-greedy action and return action we took from that
'''
def e_greedy(epsilon, Q):

    doExplore = np.random.binomial(1, epsilon)

    if doExplore:
        action = np.random.randint(k)

    else:
        # action = Q.index(max(Q))
        action = np.argmax(Q)

    return action

"""
From the given action, return the reward for action in current state.
Reward from the 10-armed testbed Rt is selected from
    normal distribution:
        mean q*(At)
        variance 1
    
"""
def reward(action_index):

    reward_mean = q[action_index]
    variance = 1
    stdv = np.sqrt(variance)
    return np.random.normal(reward_mean, stdv)

'''
Returns True if action taken is equal to action with highest q value 
Otherwise returns False

This is used during analysis of performance to determine % optimal actions at a given timestep
'''
def isOptimalAction(action):
    # optimal_action = q.index(max(q))
    optimal_action = np.argmax(q)   # gets index of action of highest q
    return True if (optimal_action == action) else False

def randomWalk(x, mean, standard_deviation):
    global q   # need to use this because this function is assigning a value to q, but we want it to point to the global q and not create a local q
    # random_walk = np.random.normal(mean, standard_deviation, len(q))

    walk_set = [-1, 0, 1]
    for i in range(len(x)):
        x[i] += np.random.choice(walk_set)

def main():
    # Perform modified 10 armed bandit testbed
    NUM_RUNS = 100
    for i in tqdm(range(NUM_RUNS), desc = "Runs", unit="run"):

        initialize_bandit_problem()

        for t in range(TIME_STEPS):
            
            # Shift true value distribution
            randomWalk(q, RANDOM_WALK_MEAN, RANDOM_WALK_STANDARD_DEVIATION)

            # Method 1: sample average
            A_1 = e_greedy(epsilon, Q[0])
            R_1 = reward(A_1)
            N[0][A_1] += 1
            Q[0][A_1] = Q[0][A_1] + ((1/N[0][A_1])*(R_1 - Q[0][A_1]))
            R_average[0][t] += R_1
            num_optimal_actions[0][t] += int(isOptimalAction(A_1))

            # Method 2: constant step size
            A_2 = e_greedy(epsilon, Q[1])
            R_2 = reward(A_2)
            N[1][A_2] += 1
            Q[1][A_2] = Q[1][A_2] + ((alpha)*(R_2 - Q[1][A_2]))
            R_average[1][t] += R_2
            num_optimal_actions[1][t] += int(isOptimalAction(A_2))

            
        
    # Average results of run
    R_average /= NUM_RUNS
    num_optimal_actions = (num_optimal_actions / NUM_RUNS) * 100   # convert to %


    # Average Reward plot
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].plot(R_average[0], label='Sample Average')
    axs[0].plot(R_average[1], label='Constant Step Size α=0.1')
    axs[0].set_xlabel('Steps')
    axs[0].set_ylabel('Average Reward')
    axs[0].set_title('Average Reward over Time')
    axs[0].legend()
    axs[0].grid(True)

    # % Optimal Action plot
    axs[1].plot(num_optimal_actions[0], label='Sample Average')
    axs[1].plot(num_optimal_actions[1], label='Constant Step Size α=0.1')
    axs[1].set_xlabel('Steps')
    axs[1].set_ylabel('% Optimal Action')
    axs[1].set_title('Optimal Action Percentage')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

# This ensures that this file is only ran when the python file is run
# Without this, if we import this python file to another program it will run it and we dont want that
if __name__ == "__main__":
    main()


