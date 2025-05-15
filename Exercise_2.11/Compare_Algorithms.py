
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
import sys
sys.path.insert(1, 'D:/Files/Desktop/Git Repo/Reinforcement_Learning/Exercise_2.5')
import e_greedy

import enum
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

class Method(enum.Enum):
    E_GREEDY = 0
    # TODO: Add more enums as we need them to index the arrays

NUM_RUNS = 10      # Ideally want 1000 or 2000 like what Exercise 2.5 wanted
NUM_STEPS = 1000    # want 200_000
LAST_STEPS_FRACTION = 1 - (0.5)       # Record this last fraction of timesteps
LAST_STEPS = NUM_RUNS * LAST_STEPS_FRACTION    # start recording rewards after this many time steps have passed

start = 1/128
end = 4
num = 1
params = np.geomspace(start, end, num)

for param in params:
    for i in range(NUM_RUNS):

        e_greedy.initialize_bandit_problem()

        # for n in range(NUM_STEPS):
            

        




