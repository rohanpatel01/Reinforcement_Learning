
import numpy as np

from Exercise_5_12.Toy_Test import vertical_increment, horizontal_increment

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
# wind = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


GOAL_ROW = 3
GOAL_COL = 7

def get_state_index_from_row_col(row, col):
    return row*NUM_COLS + col

def get_row_col_from_state_index(state_value):
    row = state_value // NUM_COLS
    col = state_value % NUM_COLS
    return row, col

TERMINAL_STATE = get_state_index_from_row_col(GOAL_ROW, GOAL_COL)

def initialize_Q():
    Q = np.empty((len(states), len(actions)), dtype=float)

    for state_index in range(len(states)):
        for action_index in range(len(actions)):
            Q[state_index][action_index] = np.random.uniform(-8, -7)

    for action_index in range(len(actions)):
        Q[TERMINAL_STATE][action_index] = 0

    return Q

def get_epsilon_greedy_action(Q, S, epsilon):

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

    row, col = get_row_col_from_state_index(S)
    vertical_action_delta, horizontal_action_delta = actions[A]
    vertical_wind_delta = wind[col]

    # - vertical wind delta bc wind makes agent move upward, which subtracts from row
    row_next = row + vertical_action_delta - vertical_wind_delta
    col_next = col + horizontal_action_delta

    # If agent's actions take it outside the world and
    # since we can only move up, down, left, right, only either row or col could be out of bounds, if so we just move that one so it is in bounds
    if (row_next < 0) or (NUM_ROWS <= row_next):
        row_next = row

    if (col_next < 0) or (NUM_COLS <= col_next):
        col_next = col

    state_next = get_state_index_from_row_col(row_next, col_next)
    reward = -1

    return state_next, reward

def SARSA(step_size, gamma, epsilon, MAX_EPISODES):

    Q = initialize_Q()

    episode_index = 0

    while episode_index < MAX_EPISODES:
        print("Starting episode ", episode_index)
        # Normally we would choose initial S st every (s,a) visited infinitely via uniform random choice for starting state S
        S = 30          # start state by definition of problem
        t = 1
        A = get_epsilon_greedy_action(Q, S, epsilon)

        # Loop through each step of episode until S is terminal
        while S != TERMINAL_STATE:
            t += 1
            S_next, R = get_observation(S, A)
            A_next = get_epsilon_greedy_action(Q, S_next, epsilon)
            Q[S][A] = Q[S][A] + step_size*(R + (gamma*Q[S_next][A_next]) - Q[S][A])
            S = S_next
            A = A_next

        episode_index += 1
        print("Completed episode ", episode_index, " with ", t, " time steps")
        print()

    return Q

def visualize_deterministic_policy(policy):

    gridworld = np.zeros(shape=(NUM_ROWS, NUM_COLS))

    step = 1
    s = 30

    row,col = get_row_col_from_state_index(s)
    gridworld[row][col] = step
    step += 1

    # not guaranteed to terminate tho?
    while s != TERMINAL_STATE:

        s, r = get_observation(s, policy[s])

        row, col = get_row_col_from_state_index(s)
        gridworld[row][col] = step
        step += 1

    print("Grid world with policy visualized: ")
    for row in range(NUM_ROWS):
        for col in range(NUM_COLS):
            print(gridworld[row][col], end=" ")

        print()
    print()


def main():

    MAX_EPISODES = 1e4

    epsilon = 0.1
    step_size = 0.5
    gamma = 1

    Q = SARSA(step_size=step_size, gamma=gamma, epsilon=epsilon, MAX_EPISODES=MAX_EPISODES)

    optimal_policy = np.empty(shape=len(states), dtype=int)
    for state_index in range(len(states)):
        optimal_policy[state_index] = Q[state_index].argmax()

    print("---Done learning from ", MAX_EPISODES, " episodes---")
    print()

    visualize_deterministic_policy(optimal_policy)


if __name__ == "__main__":
    main()

















