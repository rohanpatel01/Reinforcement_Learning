

import numpy as np
import itertools

MAX_CARS_IN_LOCATION = 20
NUM_STATES = (MAX_CARS_IN_LOCATION + 1) * 2       # plus 1 to include zero
NUM_ACTIONS = 11 # (0 to 5) and (-1 to -5)

RETURN_LAMBDA_A = 3     # max = 8
REQUEST_LAMBDA_A = 3    # max = 8

RETURN_LAMBDA_B = 2     # max = 6
REQUEST_LAMBDA_B = 4    # max = 10

def compute_p():
    pass

# compute joint probability of transitioning from current state to next state
def next_state_probability(current_state, next_state, policy):

    joint_prob = 0




    # go over all reasonable values reqA, reqB, retA, retB can take and sum their prob of getting to next state
    # 99.9% of the probability mass lies within 0 to lambda + 4*sqrt(lambda)

    reqA_range = range(0, int(REQUEST_LAMBDA_A+(4*np.sqrt(REQUEST_LAMBDA_A))))
    reqB_range = range(0, int(REQUEST_LAMBDA_B+(4*np.sqrt(REQUEST_LAMBDA_B))))

    retA_range = range(0, int(RETURN_LAMBDA_A+(4*np.sqrt(RETURN_LAMBDA_A))))
    retB_range = range(0, int(RETURN_LAMBDA_B+(4*np.sqrt(RETURN_LAMBDA_B))))

    # for retA, retB, reqA, reqB in itertools.product(reqA_range, reqB_range, retA_range, retB_range):



    pass

# for this problem we won't define an env because we will compute p for every s,a within policy eval
def policy_evaluation(policy, values, theta, gamma ):

    delta = float('inf')
    cars_A_range = range(MAX_CARS_IN_LOCATION + 1)
    cars_B_range = range(MAX_CARS_IN_LOCATION + 1)

    return_A_range = range(0, int(RETURN_LAMBDA_A + (3 * np.sqrt(RETURN_LAMBDA_A))))
    request_A_range = range(0, int(REQUEST_LAMBDA_A + (3 * np.sqrt(REQUEST_LAMBDA_A))))

    return_B_range = range(0, int(RETURN_LAMBDA_B + (3 * np.sqrt(RETURN_LAMBDA_B))))
    request_B_range = range(0, int(REQUEST_LAMBDA_B + (3 * np.sqrt(REQUEST_LAMBDA_B))))

    while not (delta < theta):

        delta = 0

        for cars_A, cars_B in itertools.product(cars_A_range, cars_B_range):
            for cars_A_next, cars_B_next in itertools.product(cars_A_range, cars_B_range):

                v = values[cars_A][cars_B]
                values[cars_A][cars_B] = 0      # set to zero for new computation
                action_A, action_B = actions[policy[cars_A][cars_B]]
                expected_p = 0.0     # compute expected probability of s'
                expected_r = 0.0     # compute expected reward from s->s'

                for returnA, requestA, returnB, requestB in itertools.product(return_A_range, request_A_range, return_B_range, request_B_range):
                    expected_p += np.random.poisson(returnA)*np.random.poisson(requestA)*np.random.poisson(returnB)*np.random.poisson(requestB)
                    expected_r += (-2*np.abs())





def policy_iteration(policy, values):
    pass


if __name__ == '__main__':

    # deterministic policy so policy just gives us the action to take in the given state
    policy = np.zeros(shape=(NUM_STATES, NUM_STATES), dtype=int) # Policy maps states [(Cars in A, Cars in B)] => Action index
    values = np.zeros(shape=(NUM_STATES, NUM_STATES), dtype=float)
    actions = []                    # Actions contains all pairs of valid actions, accessed by index given by policy

    for action_A, action_B in itertools.product(range(-5, 6), repeat=2):
        if action_A + action_B == 0:
            actions.append((action_A, action_B))

    actions.sort(key=lambda x: (x[0] < 0, abs(x[0])))
    actions = np.array(actions, dtype=int)

    new_values = policy_evaluation(policy, values, 2, 0.9)


