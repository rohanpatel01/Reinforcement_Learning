

'''
Policy iteration to re-solve Jack's car rental problem

First I will replicate results for original problem

Second I will add changes as dictated by exercise 4.7

'''

import numpy as np
import itertools

'''
Notes for original Jack's Car Rental Problem

Jack will rent out all cars available at a location and get +10 for each one
Jack has the option to take action to move cars but at reward of -2*num cars moved

+$10 for renting out a car
-$2/car moved overnight

Number of cars requested at each location: Poisson RV with lambda = 3 (first) 4 (second) location
Returned:  Poisson RV with lambda = 3 (first) 2 (second) location

Max 20 cars at each location
gamma = 0.9
continuous finine MDP:
- time steps = days
- state = [num cars location 1 end of day, num cars location 2 end of day]
- actions = [net num cars moved from/to location 1, net num cars moved from/to location 2]

from: - #
to:   + #
'''

MAX_CARS_IN_LOCATION = 20
NUM_STATES = (MAX_CARS_IN_LOCATION + 1) * 2       # plus 1 to include zero
NUM_ACTIONS = 11 # (-5 to -1) and (0 to 5)

REQUEST_LAMBDA_A = 3
REQUEST_LAMBDA_B = 4

RETURN_LAMBDA_A = 3
RETURN_LAMBDA_B = 2



"""
# max 5 cars can be moved from one location to other

# This action can be taken from any state and considers moving cars between both locations
A0 = +0
A1 = +1
...
A5 = +5

A6 = -1 
...
A10 = -5
"""


def compute_p():
    pass

# compute joint probability of transitioning from current state to next state
def next_state_probability(current_state, next_state, policy):

    joint_prob = 0




    # go over all reasonable values reqA, reqB, retA, retB can take and sum their prob of getting to next state
    # 99.9% of the probability mass lies within 0 to lambda + 4*sqrt(lambda)

    reqA_range = range(0, REQUEST_LAMBDA_A+(4*np.sqrt(REQUEST_LAMBDA_A)))
    reqB_range = range(0, REQUEST_LAMBDA_B+(4*np.sqrt(REQUEST_LAMBDA_B)))

    retA_range = range(0, RETURN_LAMBDA_A+(4*np.sqrt(RETURN_LAMBDA_A)))
    retB_range = range(0, RETURN_LAMBDA_B+(4*np.sqrt(RETURN_LAMBDA_B)))

    for retA, retB, reqA, reqB in itertools.product(reqA_range, reqB_range, retA_range, retB_range):



    pass

# for this problem we won't define an env because we will compute p for every s,a within policy eval
def policy_evaluation(policy, values, theta, gamma ):

    delta = float('inf')

    while not (delta < theta):
        delta = 0

        cars_A_range = range(MAX_CARS_IN_LOCATION + 1)
        cars_B_range = range(MAX_CARS_IN_LOCATION + 1)

        # loop through all states, to compute value of a state, to get p: only consider the next states that are likely

        # Loop over all states s
        for cars_A, cars_B in itertools.product(cars_A_range, cars_B_range):

            # Loop over all next states s'
            for cars_A_next, cars_B_next in itertools.product(cars_A_range, cars_B_range):

                # check if this state is possible
                # ^^^ if looping over all possible values of RVs do not result in next state then it is not possible and continue, else continue update computation

                p = next_state_probability((cars_A, cars_B), (cars_A_next, cars_B_next), policy)
                if p == 0: continue

                # state is possible so continue computation and return



                # TODO: Loop over all possible next states
                '''
                Loop over all possible values I, G, J, H can take
                If they
                '''

                # TODO: Figure out again why we need to loop over all the actions
                # TODO: I think I just confused but I need a way to get Pi(s)
                for action_A, action_B in actions:
                    v = values[(cars_first_location, cars_second_location)]
                    p = compute_p()



def policy_iteration(policy, values):
    pass


if __name__ == '__main__':

    # deterministic policy so policy just gives us the action to take in the given state
    policy = np.zeros(NUM_STATES)
    values = np.zeros(NUM_STATES)
    actions = np.array()

    actions = []

    for action_A, action_B in itertools.product(range(-5, 6), repeat=2):
        if action_A + action_B == 0:
            actions.append((action_A, action_B))

    actions = np.array(actions)

    new_values = policy_evaluation(policy, values, 2, 0.9)


    # reward for a state comes from: total_num_cars*10 + (-2*cars_moved)
    # have table for all possible states and populate them?
    # env is not deterministic because values are poisson rv


















