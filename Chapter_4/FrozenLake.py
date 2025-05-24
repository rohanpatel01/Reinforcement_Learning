import gymnasium as gym
import numpy as np

'''
Use Iterative Policy Evaluation to compute the true values passed in and modifies them in place 

env.P[state][action] = 
[(probability1, next_state1, reward1, terminated1), 
(probability2, next_state2, reward2, terminated2), 
...]
'''

def print_env_description(env):

    desc = env.desc

    if isinstance(env.observation_space, gym.spaces.Discrete):
        print("  Number of possible states (positions):", env.observation_space.n)
        print("  Possible state values are integers from 0 to", env.observation_space.n - 1)

    print()

    if isinstance(env.action_space, gym.spaces.Discrete):
        print("  Number of possible actions (positions):", env.action_space.n)
        print("  Possible state values are integers from 0 to", env.action_space.n - 1)

    print()

    print("\nState to (row, col) mapping:")
    num_rows = len(desc)
    num_cols = len(desc[0])
    for r in range(num_rows):
        for c in range(num_cols):
            state_index = r * num_cols + c
            cell_type = desc[r][c]
            print(f"  State {state_index:2d} corresponds to (row={r}, col={c}) - Type: '{cell_type}'")

def policy_evaluation(values, policy, env, terminal_states, theta, gamma):

    delta = float('inf')

    while not (delta < theta):

        delta = 0

        for state in range(env.observation_space.n):
            if state in terminal_states: continue

            v = values[state]

            # Compute Vk+1(s)
            new_value = 0
            for action in range(env.action_space.n):

                for i in env.P[state][action]:

                    probability, next_state, reward, done = i
                    new_value += (policy[state][action] * probability * (reward + (gamma * values[next_state])))

            delta = max(delta, np.abs(v - new_value))
            values[state] = new_value




if __name__ == '__main__':

    # state is the current position on the lake
    # static structure of environment would be
    env = gym.make("FrozenLake-v1")
    env = env.unwrapped

    # print_env_description(env)

    # Using an equiprobable policy
    policy = np.ones( (env.observation_space.n, env.action_space.n) ) / env.action_space.n

    terminal_states = {15}
    values = np.zeros(shape=(env.observation_space.n))
    policy_evaluation(values, policy, env, terminal_states, 0.0001, 0.99)
    print(values)




