import numpy as np
import time

from fontTools.ttLib.tables.S__i_l_f import aCode_info

rows, cols = 32, 17

# Fill in track with 1 where it is a valid location for car to be in
track = np.zeros((rows,cols))
track[0:6, 10::] = 1
track[0:7, 9] = 1
track[0::, 3:9] = 1
track[1:29, 2] = 1
track[3:22, 1] = 1
track[4:14, 0] = 1


# Changes probabilities of prob array such that they sum to 1 but their relative prob stay the same
def fix_p( p):
    if p.sum() != 1.0:
        p = p*(1./p.sum())
    return p

def get_random_start_col():
    start_col_begin = 3
    start_col_end = 8
    new_col = np.random.randint(start_col_begin, start_col_end + 1)
    return new_col

def get_state_index_from_row_col(row, col):
    return row*cols + col

def get_row_col_from_state_value(state_value):
    row = state_value // cols
    col = state_value % cols
    return row, col

actions = []
for horizontal_increment in [0, -1, 1]:
    for vertical_increment in [0, -1, 1]:
        actions.append([horizontal_increment, vertical_increment])

actions.remove([0,0])

states = [i for i in range(rows*cols)]

# Invalid state is a state_value whose (row,col) is not on the track (i.e. has track value of 0)
invalid_states = set()

for state_value in states:
    row, col = get_row_col_from_state_value(state_value)
    if track[row][col] == 0:
        invalid_states.add(state_value)


MAX_VELOCITY = 5

'''
Simply using algebra to see if line segment created by path intersects
the finish line (at col = 16) between rows 0 and 5 (inclusive)

Also checking if the new column is past the finish line
'''
def crossed_finishline(row, col, new_row, new_col):

    FINISH_LINE_COL = cols-1

    x = col
    y = row

    x_new = new_col
    y_new = new_row

    m = None
    if (x_new - x) == 0:
        m = 0
    else:
        m = ((y_new - y) / (x_new - x))

    b = y_new - (m*x_new)
    y_intersect = (m*FINISH_LINE_COL) + b

    if (0 <= y_intersect <= 5) and ( FINISH_LINE_COL < x_new ):
        return True
    else:
        return False


def crossed_boundary(new_row, new_col):

    if ((new_row < 0) or (rows <= new_row)) or ((new_col < 0) or (cols <= new_col)):
        return True

    if track[new_row][new_col] == 0:
        return True

    return False


def generate_episode(behavior_policy, noise):

    vertical_velocity = 0
    horizontal_velocity = 0
    episode = []
    row = rows-1
    col = get_random_start_col()

    while True:
        reward = -1
        state_index = get_state_index_from_row_col(row,col)
        action_index = np.random.choice(a=[i for i in range(len(actions))], p=fix_p(behavior_policy[state_index]))

        behavior_policy_vertical_increment = actions[action_index][0]
        behavior_policy_horizontal_increment = actions[action_index][1]

        #                 state (t)  action (t)         reward (t+1)
        episode.append( ( state_index, action_index, reward ) )

        horizontal_increment = 0
        vertical_increment = 0
        if np.random.binomial(n=1, p=noise):
            horizontal_increment = 0
            vertical_increment = 0
        else:
            horizontal_increment = behavior_policy_vertical_increment
            vertical_increment = behavior_policy_horizontal_increment

        vertical_velocity = max(-MAX_VELOCITY, min(MAX_VELOCITY, vertical_velocity + vertical_increment))
        horizontal_velocity = max(-MAX_VELOCITY, min(MAX_VELOCITY, horizontal_velocity + horizontal_increment))

        new_row = row + vertical_velocity
        new_col = col + horizontal_velocity

        if crossed_finishline(row, col, new_row, new_col):
            break

        if crossed_boundary(new_row, new_col):
            row = rows - 1
            col = get_random_start_col()
        else:
            row = new_row
            col = new_col

    return episode

def off_policy_MC_control(epsilon, gamma):

    Q = np.empty((len(states), len(actions)), dtype=float)
    for s in range(len(states)):
        for a in range(len(actions)):
            Q[s][a] = np.random.uniform(-10, -7)

    C = np.zeros((len(states), len(actions)), dtype=float)   # C[state_index, action_index] = Cumulative summation of weights for state_index and action_index

    target_policy = np.zeros(len(states), dtype=int)  # target_policy[state_index] = action index of greedy action wrt Q
    for s in range(len(states)):
        target_policy[s] = np.argmax(Q[s])

    STATE_INDEX_IN_EPISODE = 0
    ACTION_INDEX_IN_EPISODE = 1
    REWARD_INDEX_IN_EPISODE = 2

    i = 0

    while i < num_episodes:

        print("Starting ", i, "th episode")
        i += 1

        # Generate e-soft policy with respect to current Q values
        behavior_policy = np.empty(shape=(len(states), len(actions)), dtype=float)
        for state_index in range(len(states)):
            optimal_action_index = Q[state_index].argmax()
            for action_index in range(len(actions)):
                if action_index == optimal_action_index:
                    behavior_policy[state_index][action_index] = 1 - epsilon + (epsilon / len(actions))
                else:
                    behavior_policy[state_index][action_index] = (epsilon / len(actions))

        episode = generate_episode(behavior_policy, noise=0.0)
        print("Generated episode of length: ", len(episode))

        G = 0
        W = 1

        j = 0
        for t in range(len(episode)-1, -1, -1): # iterates from t = T-1, T-2, ... 0
            j+=1

            state_index = episode[t][STATE_INDEX_IN_EPISODE]
            action_index = episode[t][ACTION_INDEX_IN_EPISODE]
            reward = episode[t][REWARD_INDEX_IN_EPISODE]

            G = (gamma * G) + reward
            C[state_index][action_index] = C[state_index][action_index] + W
            Q[state_index][action_index] = Q[state_index][action_index] + ( (W / C[state_index][action_index])*(G - Q[state_index][action_index])  )

            target_policy[state_index] = Q[state_index].argmax()

            if action_index != target_policy[state_index]:
                break

            W = W * ( 1 / behavior_policy[state_index][action_index] )

        print("Learned from ", j, " (s,a) pairs")

    return Q, C, target_policy


if __name__ == "__main__":

    num_episodes = 1e3

    gamma = 1
    Q, C, target_policy = off_policy_MC_control(epsilon=0.4, gamma=gamma)  # 0.0 epsilon means 0 prob taking random action thus behavior policy will be greedy


    print("Actions: ", actions)
    print("Off-policy MC control complete")
    print("Q(S,A)")
    for s in range(len(states)):
        for a in range(len(actions)):
            print(Q[s][a], end=" ")

        print()

    print()
    print("Optimal target policy: ")
    for s in range(len(states)):
        print(target_policy[s])








