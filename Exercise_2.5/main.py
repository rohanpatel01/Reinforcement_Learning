
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


SEED = 42
RANDOM_WALK_MEAN = 0
RANDOM_WALK_STANDARD_DEVIATION = 0.01
k = 10  # 10 Actions are: 0 1 2 3 4 5 6 7 8 9
epsilon = 0.1
TIME_STEPS = 10_000
# np.random.seed(SEED)

q = None
Q = None
N = None
R = None
R_average = None
num_optimal_actions = None

R_average = np.array([None] * 2)
R_average[0] = np.array([0 for _ in range(TIME_STEPS)])
R_average[1] = np.array([0 for _ in range(TIME_STEPS)])

num_optimal_actions = np.array([0] * 2)


# num_optimal_actions = np.array([0] * 2)

def initialize_bandit_problem():

    global q 
    global Q
    global N 
    global R 
    global R_average
    global num_optimal_actions

    # True q*(a) values (Will use this to compute the rewards)
    q = np.array([1 for _ in range(k)]) 

    # Running value for incremental value for all actions
    Q = np.array([None] * 2)
    Q[0] = np.array([0 for _ in range(k)]) # Estimated Q for method 1
    Q[1] = np.array([0 for _ in range(k)]) # Estimated Q for method 2

    # Number of times we've seen action thus far
    N = np.array([None] * 2)
    N[0] = np.array([0 for _ in range(k)])
    N[1] = np.array([0 for _ in range(k)])

    # Variables we document for plotting
    R = np.array([None] * 2)
    R[0] = np.array([0 for _ in range(TIME_STEPS)])
    R[1] = np.array([0 for _ in range(TIME_STEPS)])

    


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


def randomWalk(mean, standard_deviation):
    global q   # need to use this because this function is assigning a value to q, but we want it to point to the global q and not create a local q
    random_walk = np.random.normal(mean, standard_deviation, len(q))
    q = q + random_walk

# Perform modified 10 armed bandit testbed
NUM_RUNS = 100
for i in tqdm(range(NUM_RUNS), desc = "Runs", unit="run"):

    initialize_bandit_problem()

    for t in range(TIME_STEPS):

        # Method 1: sample average
        A_1 = e_greedy(epsilon, Q[0])
        R_1 = reward(A_1)
        N[0][A_1] += 1
        Q[0][A_1] = Q[0][A_1] + ((1/N[0][A_1])*(R_1 - Q[0][A_1]))

        # Shift true value distribution
        randomWalk(RANDOM_WALK_MEAN, RANDOM_WALK_STANDARD_DEVIATION)

        # Document reward for plotting
        R[0][t] = R_1
        R_average[0][t] += R_1
        # num_optimal_actions[0] += 1 if isOptimalAction(A_1) else 0

        

        # Method 2: constant step size
        # TODO
    
    # Average results of run
R_average /= NUM_RUNS

### Plot results
sns.set_theme(style="whitegrid")

# Plot reward
# R = pd.DataFrame(R, columns=['Reward'])

x_axis = [n for n in range(TIME_STEPS)]

df = pd.DataFrame({
    'Steps': np.concatenate([x_axis]),          # , x_axis
    'Reward': R_average[0],                    # [ R[0], R[1] ]
    'Methods': np.concatenate([['Method 1'] * len(x_axis), ])   # ['Method 2'] * len(x_axis),       
})

# Create the scatter plot using Seaborn
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Steps', y='Reward', hue='Methods', data=df, s=5, alpha=0.6)

plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Scatter Plot of Three Things")
plt.legend(title="Thing") # Optional: Customize legend title
plt.show()



# Plot % Optimal action




# def plot_reward_distribution():
#     pass

#============================================================================================
# Sandbox to test stuff
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np

# sns.set_theme(style="whitegrid")

# action = 0
# reward_mean = q[action]
# stdv = np.sqrt(1)
# reward_distribution_array = np.random.normal(reward_mean, stdv, (200))

# reward_distribution = pd.DataFrame(reward_distribution_array, columns=['Reward'])

# f, ax = plt.subplots(figsize=(8, 6)) # Adjusted figure size
# ax.set(ylim=(-5, 5))

# sns.violinplot(data=reward_distribution, y='Reward', bw_adjust=.5, cut=1, linewidth=1, palette="Set3")
# ax.axhline(y=reward_mean, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {reward_mean:.2f}')

# sns.despine(left=True, bottom=True)
# plt.ylabel("Reward Value")
# plt.title("Distribution of Rewards")
# plt.show()