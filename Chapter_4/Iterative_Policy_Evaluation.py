
'''
Doing iterative policya evaluation on Example 4.1: 4x4 Gridworld
S = {1, 2, ... 14}
A = {up, down, right, left}, which deterministically cause the corresponding
state transitions, except that actions that would take the agent o↵ the grid in fact leave
the state unchanged.

Undiscounted
Episodic Task
- Keep in mind that we define V(T) = 0 for all episodic tasks

Agent follows the equiprobable random policy (all actions equally likely)

'''
import numpy as np
from collections import defaultdict

value = []
state = []
actions = ['up', 'down', 'left', 'right']
policy = np.random.choice(actions)

# key = (state, action)
# value = new state
transition_matrix = defaultdict(int) 

NUM_ROWS = 4
NUM_COLS = 4

# Note: using grid system to keep track of unique states, x being col, y being row

def is_valid_action(state, action):
    
    if action not in actions:
        raise Exception("Rohan Defined Exception: Unknown Action - double check spelling!")

    x, y = get_coords(state)

    if action == 'up':
        return True if (0<=(y-1)<NUM_ROWS) else False
    
    if action == 'down':
        return True if (0<=(y+1)<NUM_ROWS) else False
    
    if action == 'left':
        return True if (0<=(x-1)<NUM_COLS) else False
    
    if  action == 'right':
        return True if (0<=(x+1)<NUM_COLS) else False
    
def get_state(x, y):
    return (y*NUM_COLS) + x

def get_coords(state):
    x = (state % NUM_COLS)
    y = (state-x)/NUM_COLS
    return (x,y)

def populate_transition_matrix():

    for x in range(NUM_COLS):
        for y in range(NUM_ROWS):
            
            state = get_state(x, y)

            if is_valid_action(state, 'up'):
                transition_matrix[(state, 'up')] = get_state(x, y-1)
            else:
                transition_matrix[(state, 'up')] = state


            if is_valid_action(state, 'down'):
                transition_matrix[(state, 'down')] = get_state(x, y+1)
            else:
                transition_matrix[(state, 'down')] = state


            if is_valid_action(state, 'left'):
                transition_matrix[(state, 'left')] = get_state(x-1, y)
            else:
                transition_matrix[(state, 'left')] = state
            

            if is_valid_action(state, 'right'):
                transition_matrix[(state, 'right')] = get_state(x+1, y)
            else:
                transition_matrix[(state, 'right')] = state


def sample_policy(policy, state):
    pass

def evaluate_policy():
    pass


def run_iterative_policy_evaluation():
    pass

if __name__ is '__main__':
    run_iterative_policy_evaluation()
