import numpy as np
import torch
import torch.nn as nn


from Q_Learning import Q_Learning
from TestEnv import TestEnv
from Config.LinearConfig import LinearConfig
from Scheduler import EpsilonScheduler


class Linear(Q_Learning):

    def __init__(self, env, config):
        super(Linear, self).__init__(env, config)
        self.env = env
        self.config = config
        self.target_network = LinearNN(env, config)
        self.approx_network = LinearNN(env, config)

    # TODO
    def sample_action(self, env, state, epsilon, network_name):

        if np.random.rand() < epsilon:
            # take random action
            return np.random.randint(env.numActions)
        else:
            # select greedy action
            return self.get_best_action(state, network_name)

    def get_Q_value(self, state, action, network_name):
        # Note: action is the index of the action defined in the environment's action space
        if network_name == "target":
            Q_actions = self.target_network.forward(state)

        elif network_name == "approx":
            pass # TODO: need to implement

        return Q_actions[int(action.numpy())]

    # Returns index of action in action space with highest Q value
    def get_best_action(self, state, network_name) -> int:
        if network_name == "target":
            Q_actions = self.target_network.forward(state)
            best_action_index = np.argmax(Q_actions.detach())
        elif network_name == "approx":
            pass # TODO: need to implement

        return best_action_index


    def perform_gradient_descent(self):
        # TODO: figure out if we need autograd here and how it works / what we need to do here to get it working
        # in order to update weights based on the loss
        raise NotImplementedError

    def set_target_weights(self):
        # replace target weights with that of the current network's weights
        raise NotImplementedError


class LinearNN(nn.Module):
    def __init__(self, env, config):
        super(LinearNN, self).__init__()
        self.env = env
        self.config = config
        self.fc1 = nn.Linear(np.prod(self.env.state_shape)*self.config.frame_stack_size, self.env.numActions)
        self.double()       # converts model to double to match datatype of input with no serious performance problems

    def forward(self, x):
        x = self.fc1(x)
        return x



def q_value_test():
    env = TestEnv()
    config = LinearConfig
    model = Linear(env, config)

    for state in env.states:
        for action in env.actions:
            if env.done:
                break
            next_state, reward = env.take_action(state, action)
            experience_tuple = (state, action, reward, next_state, env.done)
            model.replay_buffer.store(experience_tuple)
        if env.done:
            break

    state, action, reward, next_state, done = model.replay_buffer.replay_buffer[0]
    Q_state_action = model.get_Q_value(state, action, "target")
    print("Q value for state: ", state, " and action: ", action, " : ", Q_state_action)

def get_best_action_test():
    env = TestEnv()
    config = LinearConfig
    model = Linear(env, config)

    for state in env.states:
        for action in env.actions:
            if env.done:
                break
            next_state, reward = env.take_action(state, action)
            experience_tuple = (state, action, reward, next_state, env.done)
            model.replay_buffer.store(experience_tuple)
        if env.done:
            break

    state, action, reward, next_state, done = model.replay_buffer.replay_buffer[0]
    # Show Q action values of state
    best_action = model.get_best_action(state, "target")
    print("best action: ", best_action)

    # Show best action

def sample_action_test():

    # torch.manual_seed(42)

    env = TestEnv()
    config = LinearConfig
    model = Linear(env, config)

    for state in env.states:
        for action in env.actions:
            if env.done:
                break
            next_state, reward = env.take_action(state, action)
            experience_tuple = (state, action, reward, next_state, env.done)
            model.replay_buffer.store(experience_tuple)
        if env.done:
            break

    state, action, reward, next_state, done = model.replay_buffer.replay_buffer[0]

    epsilon_scheduler = EpsilonScheduler(config.begin_epsilon, config.end_epsilon,
                                         config.max_time_steps_update_epsilon)

    e = .4

    total_runs = 1000000
    total_greedy = 0
    total_random = 0

    best_action = model.get_best_action(state, "target")

    print("State: ", state)
    print("Best action: ", best_action)
    print(model.target_network.forward(state))

    for i in range(total_runs):
        action = model.sample_action(env, state, e, "target")

        if action == best_action:
            total_greedy += 1
        else:
            total_random += 1

    print("Epsilon (% take random action): ", e)
    print("Greedy: ", total_greedy / total_runs, "%")
    print("Random: ", total_random / total_runs, "%")











def main():
    # env = TestEnv()
    # config = LinearConfig
    # model = Linear(env, config)
    # print("fc1: ", model.linearNN.fc1.weight)

    # q_value_test()
    # get_best_action_test()
    sample_action_test()

    # model.train()


if __name__ == '__main__':
    main()

