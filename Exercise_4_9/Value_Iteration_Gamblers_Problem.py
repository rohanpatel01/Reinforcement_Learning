
import numpy as np
import time

import matplotlib.pyplot as plt

''' 
Implement value iteration for the gambler’s problem and
solve it for ph = 0.25 and ph = 0.55. In programming, you may find it convenient to
introduce two dummy states corresponding to termination with capital of 0 and 100,
giving them values of 0 and 1 respectively. Show your results graphically, as in Figure 4.3.

Are your results stable as theta -> 0? 

# p(s', r | s, a) = Ph
# ^^^ Depends on whether Heads or tails
# possible next states are Heads => state += bet
#                          Tails => state -= bet

'''


def value_iteration(actions, rewards, theta, gamma, probability_heads):

    sweeps = []

    next_sweep_index = 1

    values = np.zeros(NUM_STATES)
    values[0] = 0
    values[NUM_STATES - 1] = 0

    delta = float('inf')
    i = 0
    while not (delta < theta):

        delta = 0

        # Don't include terminal states (s = 0 and s = 100)
        for s in states[1:NUM_STATES-1]:
            v = values[s]

            # Compute new V(s)
            max_value_a = 0

            for a in actions[s]:
                value_a = 0

                # Next state is heads
                s_next_heads = s + a
                value_a += ( (probability_heads) * (rewards[s_next_heads] + (gamma*values[s_next_heads])) )

                # Next state is tails
                s_next_tails = s - a
                value_a += ( (1 - probability_heads) * (rewards[s_next_tails] + (gamma*values[s_next_tails])))

                max_value_a = max(max_value_a, value_a)

            values[s] = max_value_a
            delta = max(delta, abs(v - values[s]))

        i += 1
        if i == next_sweep_index:
            sweeps.append( (i, np.copy(values)) )
            next_sweep_index *= 2

    # End of While Loop

    # Get deterministic policy
    policy = np.empty(NUM_STATES)

    for s in states:

        argmax_a = None
        policy_max_value_a = 0

        for a in actions[s]:
            value_a = 0

            # Next state is heads
            s_next_heads = s + a
            value_a += ((probability_heads) * (rewards[s_next_heads] + (gamma * values[s_next_heads])))

            # Next state is tails
            s_next_tails = s - a
            value_a += ((1 - probability_heads) * (rewards[s_next_tails] + (gamma * values[s_next_tails])))

            if value_a > policy_max_value_a:
                policy_max_value_a = value_a
                argmax_a = a

        policy[s] = argmax_a


    return policy, values, sweeps

def plot_results():

    capital = np.arange(start=1, stop=100, step=1)

    # Plot Values vs Capital
    plt.figure()

    for i, sweep in enumerate(sweeps):
        label = "Sweep: " + str(sweep[0]+1)
        plt.step(capital, sweep[1][1: NUM_STATES-1], drawstyle='steps', label=label)

    plt.xlabel("Capital")
    plt.ylabel("Values").set_rotation(0)
    plt.legend(loc='best')
    plt.show()


    # Plot Final policy (stake) vs Capital
    plt.figure()
    plt.step(capital, policy[1: NUM_STATES-1], drawstyle='steps')
    plt.xlabel("Capital")
    plt.ylabel("Final Policy").set_rotation(0)
    plt.show()






if __name__ == '__main__':

    # 0.4 because want to recreate the figure they gave
    probability_heads = 0.4                                    # TODO: Also try with Ph = 0.55 and compare after algorithm works

    NUM_STATES = 100 + 1
    gamma = 1 # undiscounted case
    theta = 1e-12

    print("Initializing")
    states = np.arange(start=0, stop=NUM_STATES, step=1, dtype=int)
    rewards = np.zeros(NUM_STATES)
    rewards[NUM_STATES-1] = 1                                            # Goal state has reward of +1 and all else have reward of 0

    # Populate action space for every state
    actions = np.empty(shape=(NUM_STATES), dtype=object)
    for s in states:
        actions[s] = np.arange(start=0, stop=min(s, (NUM_STATES-1) - s ) + 1, step=1, dtype=int)

    print("Starting value iteration")
    start_time = time.time()
    policy, values, sweeps = value_iteration(actions, rewards, theta, gamma, probability_heads)
    print("--- %s seconds to execute value_iteration ---" % (time.time() - start_time))

    plot_results()





    # print("=============================")
    # print("Output: \n")
    # print("Values: ")
    # for s in states:
    #     print("State ",s, ": ", values[s])
    #
    # print()
    #
    # print("Policy: ")
    # for s in states:
    #     print("State ", s, ": ", policy[s])










