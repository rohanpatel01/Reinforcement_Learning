

import numpy as np
import itertools
from scipy.stats import poisson
import matplotlib.pyplot as plt


MAX_CARS_IN_LOCATION = 10
NUM_STATES = MAX_CARS_IN_LOCATION + 1       # plus 1 to include zero
NUM_ACTIONS = 11 # (0 to 5) and (-1 to -5)

RETURN_LAMBDA_A = 3     # max = 8
REQUEST_LAMBDA_A = 3    # max = 8

RETURN_LAMBDA_B = 2     # max = 6
REQUEST_LAMBDA_B = 4    # max = 10


# for this problem we won't define an env because we will compute p for every s,a within policy eval
def policy_evaluation(policy, values, actions, theta, gamma ):

    print("Starting Policy Evaluation")

    delta = float('inf')
    cars_A_range = range(MAX_CARS_IN_LOCATION + 1)
    cars_B_range = range(MAX_CARS_IN_LOCATION + 1)

    return_A_range = range(0, int(RETURN_LAMBDA_A + (3 * np.sqrt(RETURN_LAMBDA_A))) - 2)
    request_A_range = range(0, int(REQUEST_LAMBDA_A + (3 * np.sqrt(REQUEST_LAMBDA_A))) - 2)

    return_B_range = range(0, int(RETURN_LAMBDA_B + (3 * np.sqrt(RETURN_LAMBDA_B))) - 2)
    request_B_range = range(0, int(REQUEST_LAMBDA_B + (3 * np.sqrt(REQUEST_LAMBDA_B))) - 2)

    i = 0

    while not (delta < theta):

        delta = 0

        for cars_A, cars_B in itertools.product(cars_A_range, cars_B_range):

            v = values[cars_A][cars_B]
            new_value = 0
            # values[cars_A][cars_B] = 0      # set to zero for new computation
            action_A, action_B = actions[policy[cars_A][cars_B]]

            # loop over all possible next state as dictated by likely values that return and request and using them to compute next state
            for return_A, request_A, return_B, request_B in itertools.product(return_A_range, request_A_range, return_B_range, request_B_range):

                # Ensure action is valid for state
                #   case: we don't have enough cars to give away    case: we dont have enough spaces to take in cars
                if ( (action_A < 0) and (cars_A < abs(action_A))) or ( (action_A > 0) and (MAX_CARS_IN_LOCATION - cars_A) < action_A ):
                    continue

                cars_A_after_action = cars_A + action_A
                cars_B_after_action = cars_B + action_B

                rent_A = min(cars_A_after_action, request_A)
                rent_B = min(cars_B_after_action, request_B)

                next_A = min(MAX_CARS_IN_LOCATION, cars_A_after_action - rent_A + return_A)
                next_B = min(MAX_CARS_IN_LOCATION, cars_B_after_action - rent_B + return_B)

                probability = (
                    poisson.pmf(return_A, RETURN_LAMBDA_A) *
                    poisson.pmf(request_A, REQUEST_LAMBDA_A) *
                    poisson.pmf(return_B, RETURN_LAMBDA_B) *
                    poisson.pmf(request_B, REQUEST_LAMBDA_B)
                )

                # Note: Just using action_A to represent number of cars moved overnight bc action_B is just the inverse
                reward = 10*(rent_A + rent_B) -2*abs(action_A) # TODO: This is where improper action could affect V(S)
                new_value += (probability * (reward + (gamma * values[next_A][next_B])))

            # end of next state
            delta = max(delta, abs(v - new_value))
            values[cars_A][cars_B] = new_value
            # print("State: ", cars_A ," ", cars_B)
            # print("Delta: ", delta)
            # print("theta: ", theta)
            # print("-------------------------")
        # end of current state

        i += 1
        print("i: ", i, " delta: ", delta)

    # end of while loop
    return values

def policy_improvement(policy, values, actions, gamma):

    print("Starting Policy Improvement")
    cars_A_range = range(MAX_CARS_IN_LOCATION + 1)
    cars_B_range = range(MAX_CARS_IN_LOCATION + 1)

    return_A_range = range(0, int(RETURN_LAMBDA_A + (3 * np.sqrt(RETURN_LAMBDA_A))) - 2)
    request_A_range = range(0, int(REQUEST_LAMBDA_A + (3 * np.sqrt(REQUEST_LAMBDA_A))) - 2)

    return_B_range = range(0, int(RETURN_LAMBDA_B + (3 * np.sqrt(RETURN_LAMBDA_B))) - 2)
    request_B_range = range(0, int(REQUEST_LAMBDA_B + (3 * np.sqrt(REQUEST_LAMBDA_B))) - 2)

    policy_stable = True
    for cars_A, cars_B in itertools.product(cars_A_range, cars_B_range):

        old_action_index = policy[cars_A][cars_B]
        old_action_A, old_action_B = actions[policy[cars_A][cars_B]]

        highest_actions = []
        highest_value = float('-inf')

        # compute argmax a  V(s)
        for i in range(len(actions)):
            action_A, action_B = actions[i]

            # Compute value of state from following current action
            value = 0
            for return_A, request_A, return_B, request_B in itertools.product(return_A_range, request_A_range, return_B_range, request_B_range):

                # Ensure action is valid for state
                #   case: we don't have enough cars to give away    case: we dont have enough spaces to take in cars
                if ( (action_A < 0) and (cars_A < abs(action_A))) or ( (action_A > 0) and (MAX_CARS_IN_LOCATION - cars_A) < action_A ):
                    continue

                cars_A_after_action = cars_A + action_A
                cars_B_after_action = cars_B + action_B

                rent_A = min(cars_A_after_action, request_A)
                rent_B = min(cars_B_after_action, request_B)

                next_A = min(MAX_CARS_IN_LOCATION, cars_A_after_action - rent_A + return_A)
                next_B = min(MAX_CARS_IN_LOCATION, cars_B_after_action - rent_B + return_B)

                probability = (
                    poisson.pmf(return_A, RETURN_LAMBDA_A) *
                    poisson.pmf(request_A, REQUEST_LAMBDA_A) *
                    poisson.pmf(return_B, RETURN_LAMBDA_B) *
                    poisson.pmf(request_B, REQUEST_LAMBDA_B)
                )

                # Note: Just using action_A to represent number of cars moved overnight bc action_B is just the inverse
                reward = 10*(rent_A + rent_B) -2*abs(action_A) # TODO: This is where improper action could affect V(S)
                value += (probability * (reward + (gamma * values[next_A][next_B])))

            if value > highest_value:
                highest_value = value
                highest_actions = [i]

            elif value == highest_value:
                highest_actions.append(i)


        # choose the first action in highest_actions as to avoid switching between actions that are equally good
        policy[cars_A][cars_B] = highest_actions[0]

        if old_action_index not in highest_actions:
            policy_stable = False
    # end of for each s

    return policy_stable, policy

    # if policy_stable:
    #     return values, policy
    # else:
    #     new_values = policy_evaluation(policy, values, 10, 0.9)

    # TODO: After we get policy improvement to work (can check by running it once and ensuring that the value for every state increases)
    # TODO:

def generate_policy_graph(policy, actions):
    # Extract the action_A component for each state
    policy_actions = np.array([[actions[policy[j, i]][0] for i in range(policy.shape[1])] for j in range(policy.shape[0])])

    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    # Create a custom colormap and normalization for the discrete actions
    # Actions range from -4 to 5 based on the image example.
    # We need a color for each unique action.
    unique_actions = np.sort(np.unique(policy_actions))
    num_actions = len(unique_actions)

    # Create a colormap. You can customize these colors.
    # For now, let's use a standard diverging colormap that goes from blue (negative) to red (positive)
    # with white in the middle for 0.
    cmap = plt.cm.RdBu_r # Red-Blue reversed, so red for positive, blue for negative.
    if 0 in unique_actions:
        # Adjust colormap to ensure 0 is centered, e.g., with white
        # This is more complex if using a standard colormap.
        # A simpler approach for distinct colors is to define them manually.
        colors = plt.cm.get_cmap('Spectral', num_actions) # Or 'tab10', 'viridis', etc.
        # Alternatively, define colors explicitly for distinct actions for better control
        # Example: { -4: 'darkblue', -3: 'blue', -2: 'lightblue', -1: 'cyan',
        #            0: 'white',
        #            1: 'lightcoral', 2: 'salmon', 3: 'red', 4: 'darkred', 5: 'maroon' }
        # Map unique_actions to a range [0, num_actions-1] for colormap indexing
        norm = plt.Normalize(vmin=unique_actions.min(), vmax=unique_actions.max())
    else:
        norm = plt.Normalize(vmin=unique_actions.min(), vmax=unique_actions.max())
        colors = plt.get_cmap('viridis', num_actions) # Example for general case

    # Using pcolormesh to create the colored squares.
    # X and Y define the boundaries of the cells.
    # We need to add 1 to the max to ensure the last cell is included.
    # The image has y-axis (cars at first location) from 0 to 20, and x-axis (cars at second location) from 0 to 20.
    # policy_actions[j, i] corresponds to cars_A=j, cars_B=i.
    # For pcolormesh, (X, Y) are the coordinates of the vertices.
    # So if you have N states in X and M states in Y, you need (N+1, M+1) vertices.
    mesh = ax.pcolormesh(np.arange(MAX_CARS_IN_LOCATION + 1),
                         np.arange(MAX_CARS_IN_LOCATION + 1),
                         policy_actions,
                         cmap='RdBu_r', # Diverging colormap is good for actions +/-
                         edgecolors='gray', linewidth=0.5,
                         norm=plt.Normalize(vmin=policy_actions.min(), vmax=policy_actions.max()))

    # Add text labels for each cell
    for i in range(MAX_CARS_IN_LOCATION + 1):
        for j in range(MAX_CARS_IN_LOCATION + 1):
            action = policy_actions[j, i] # j is row (Cars A), i is column (Cars B)
            ax.text(i + 0.5, j + 0.5, f"{int(action)}", # Centered in the square
                    ha='center', va='center', color='black', fontsize=8)


    ax.set_xlabel('#Cars at second location', fontsize=12)
    ax.set_ylabel('#Cars at first location', fontsize=12)
    ax.set_title(r'$\pi_3$', fontsize=16)

    # Set ticks and limits. Ensure limits go from 0 to MAX_CARS_IN_LOCATION.
    # Ticks should be at 0 and 20.
    ax.set_xticks([0, MAX_CARS_IN_LOCATION])
    ax.set_yticks([0, MAX_CARS_IN_LOCATION])

    ax.set_xlim(0, MAX_CARS_IN_LOCATION + 1) # +1 because pcolormesh takes vertices
    ax.set_ylim(0, MAX_CARS_IN_LOCATION + 1)

    # Invert the y-axis to match the image
    ax.invert_yaxis()

    # Create a colorbar
    cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.05)
    cbar.set_label('Action (Cars Moved A to B)')
    cbar.set_ticks(unique_actions) # Set ticks to the exact action values

    plt.grid(False) # No grid lines, as in the example image
    plt.tight_layout()
    plt.show()


def policy_iteration(policy, values, actions, theta, gamma):
    # TODO: Change iteration to be forever until convergence to optimal
    # TODO: Make sure to continue printing to ensure we know what's going on in the loop - so ADD MORE PRINT STATEMENTS
    i = 0
    while True:
        # 2. Policy Evaluation
        values = policy_evaluation(policy, values, actions, theta, gamma)

        print("-----------")
        print("Values : (They should be strictly increasing unless already optimal)")
        print(values)
        print("-----------")

        # 3. Policy Improvement
        policy_stable, policy = policy_improvement(policy, values, actions, gamma)

        print("Policy: ")
        for A in range(0, MAX_CARS_IN_LOCATION + 1):
            for B in range(0, MAX_CARS_IN_LOCATION + 1):
                print(actions[policy[A][B]][0], end=" ")
            print()


        # Note: if policy is stable then the value of this policy will be the same as before so no need to recompute the value of this (now optimal) policy bc we already have
        # it from the previous iteration and it will be "values"

        if policy_stable:
            return values, policy
        else:
            i += 1
            print("Policy Iteration: i: ", i)



    return values, policy







if __name__ == '__main__':

    # 1. Initialization
    policy = np.zeros(shape=(NUM_STATES, NUM_STATES),
                      dtype=int)  # Policy maps states [(Cars in A, Cars in B)] => Action index
    values = np.full(shape=(NUM_STATES, NUM_STATES), fill_value=100.0,
                     dtype=float)  # optimistic initialization to get faster convergence and encourage exploration
    actions = []  # Actions contains all pairs of valid actions, accessed by index given by policy

    for action_A, action_B in itertools.product(range(-5, 6), repeat=2):
        if action_A + action_B == 0:
            actions.append((action_A, action_B))

    actions.sort(key=lambda x: (x[0] < 0, abs(x[0])))
    actions = np.array(actions, dtype=int)

    theta = 0.1
    gamma = 0.9


    # deterministic policy so policy just gives us the action to take in the given state
    values, policy = policy_iteration(policy, values, actions, theta, gamma)

    print("=====================================")
    print("=====================================")

    print("Final ~Optimal Values and Policy")
    # generate_policy_graph(policy, actions)

    print("Policy: ")
    for A in range(0, MAX_CARS_IN_LOCATION + 1):
        for B in range(0, MAX_CARS_IN_LOCATION + 1):
            print(actions[policy[A][B]][0], end=" ")
        print()

    print("Values: ")
    print(values)

