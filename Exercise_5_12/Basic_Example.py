import numpy as np
import time


states = [0, 1, 2, 3, 4, 5]
actions = [1, -1]

def generate_episode(behavior_policy):

    state_index = 0
    episode = []

    while True:
        reward = -1
        action_index = np.random.choice(a = [0, 1], p=behavior_policy[state_index])

        episode.append([state_index, action_index, -1])

        state_index += actions[action_index]
        if state_index < 0:
            state_index = 0
        elif state_index == 5:
            break


    return episode


def off_policy_MC_control(epsilon, theta, gamma):
    # Q = np.full( (len(states), len(actions)), fill_value=-99, dtype=float)    # Q[state_index][action_index] = value of (s,a) pair

    Q = np.empty((len(states), len(actions)), dtype=float)

    for s in range(len(states)):
        for a in range(len(actions)):
            Q[s][a] = np.random.uniform(-10,  -7)


    C = np.zeros((len(states), len(actions)), dtype=float)      # C[state_index][action_index] = Cummilative sum of weights for (s,a) pair
    target_policy = np.zeros(len(states), dtype=int)    # target_policy[state_index] = action_index

    for s in range(len(states)):
        target_policy[s] = np.argmax(Q[s])

    behavior_policy = np.zeros((len(states), len(actions)))            # behavior_policy[state_index][action_index] = prob of taking action @ index in state @ index

    print("Initial Values: ")
    print("Q values")
    for state_index in range(len(states)):
        for action_index in range(len(actions)):
            print("State: ", state_index, " Action: ", actions[action_index], " Value: ", Q[state_index][action_index])

    print()
    print("Optimal target policy: ")
    for s in range(len(states)):
        print(target_policy[s])

    time.sleep(5)


    i = 0
    while i < MAX_EPISODES: # not (delta < theta)

        # delta = 0

        print("Episode: ", i + 1)
        i += 1

        # e-soft behavior policy
        for state_index in range(len(states)):
            optimal_action_index = Q[state_index].argmax()

            for action_index in range(len(actions)):
                if action_index == optimal_action_index:
                    behavior_policy[state_index][action_index] = 1 - epsilon + (epsilon/len(actions))
                else:
                    behavior_policy[state_index][action_index] = (epsilon / len(actions))

        episode = generate_episode(behavior_policy)
        print("Len episode: ", len(episode))

        G = 0
        W = 1
        for t in range(len(episode)-1, -1, -1):

            state_index = episode[t][0]
            action_index = episode[t][1]
            reward = episode[t][2]

            # before = Q[state_index][action_index]

            G = (gamma*G) + reward
            C[state_index][action_index] = C[state_index][action_index] + W
            Q[state_index][action_index] = Q[state_index][action_index] + ( (W/C[state_index][action_index])*(G - Q[state_index][action_index])  )
            target_policy[state_index] = Q[state_index].argmax()

            # delta = max(delta, abs(before - Q[state_index][action_index]))

            # print("State: ", state_index, " Action: ", actions[action_index], " Delta: ", delta)
            # print("===================")
            if action_index != target_policy[state_index]:
                break

            W = W * (1/behavior_policy[state_index][action_index])


    return Q, C, target_policy


if __name__ == "__main__":

    MAX_EPISODES = 1e3

    theta = 0.1
    gamma = 1

    Q, C, target_policy = off_policy_MC_control(epsilon=0.7, theta=theta,
                                                gamma=gamma)  # 0.0 epsilon means 0 prob taking random action thus behavior policy will be greedy


    print("Actions: ", actions)

    print("Q values")
    for state_index in range(len(states)):
        for action_index in range(len(actions)):
            print("State: ", state_index, " Action: ", actions[action_index], " Value: ", Q[state_index][action_index])

    print()
    print("Optimal target policy: ")
    for s in range(len(states)):
        print(target_policy[s])


