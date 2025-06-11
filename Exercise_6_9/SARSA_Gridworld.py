
import numpy as np

# Indices for accessing elements of action
VERTICAL_ACTION = 0
HORIZONTAL_ACTION = 1

NUM_ROWS = 7
NUM_COLS = 10

states = [s for s in range(NUM_ROWS*NUM_COLS)]

# (vertical_action, horizontal_action)
#             up     down    right   left
actions = [(-1, 0), (1, 0), (0, 1), (0, -1)]

print("Actions: \n", actions, end="\n\n")

print("States: ")
for row in range(NUM_ROWS):
    for col in range(NUM_COLS):
        print(states[row*NUM_COLS + col], end=" ")

    print()
print()

wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

GOAL_ROW = 3
GOAL_COL = 7

def get_state_index_from_row_col(row, col):
    return row*NUM_COLS + col

def get_row_col_from_state_value(state_value):
    row = state_value // NUM_COLS
    col = state_value % NUM_COLS
    return row, col


def main():

    MAX_EPISODES = 1e1
    step_size = 0.5
    gamma = 1

    SARSA(step_size, gamma, MAX_EPISODES)

def initialize_Q():
    Q = np.empty((len(states), len(actions)), dtype=float)

    for state_index in range(len(states)):
        for action_index in range(len(actions)):
            Q[state_index][action_index] = np.random.uniform(-10, -7)

    TERMINAL_STATE = get_state_index_from_row_col(GOAL_ROW, GOAL_COL)
    for action_index in range(len(actions)):
        Q[TERMINAL_STATE][action_index] = 0

    return Q

def get_epsilon_greedy_action(Q, S, t):

    epsilon = 1/t

    probabilities = [0 for i in range(len(actions))]
    choices = [action_index for action_index in range(len(actions))]

    optimal_action_index = Q[S].argmax()

    for action_index in choices:
        if action_index == optimal_action_index:
            probabilities[action_index] = 1 - epsilon + (epsilon/len(actions))
        else:
            probabilities[action_index] = (epsilon/len(actions))

    return np.random.choice(a=choices, p=probabilities)

def get_observation(S, A):
    # TODO: Implement this
    pass


def SARSA(step_size, gamma, MAX_EPISODES):

    Q = initialize_Q()

    episode_index = 0

    while episode_index < MAX_EPISODES:
        # Normally we would choose initial S st every (s,a) visited infinitely via uniform random choice for starting state S
        S = 30          # start state by definition of problem
        t = 1           # t determines
        A = get_epsilon_greedy_action(Q, S, t)

        # Loop through each step of episode until S is terminal
        while True:
            S_next, R = get_observation(S, A)


            t += 1


        episode_index += 1







if __name__ == "__main__":
    main()

















