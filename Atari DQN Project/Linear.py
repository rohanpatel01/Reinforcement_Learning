import numpy as np
import torch
import torch.nn as nn
from torch import optim


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
            Q_actions = self.approx_network.forward(state)

        # action here is an int and not a tensor so we don't need to convert it
        return Q_actions[action] # int(action.numpy())

    # Returns index of action in action space with highest Q value
    def get_best_action(self, state, network_name) -> int:
        if network_name == "target":
            Q_actions = self.target_network.forward(state)
        elif network_name == "approx":
            Q_actions = self.approx_network.forward(state)

        best_action_index = np.argmax(Q_actions.detach())
        return best_action_index

    def perform_gradient_descent(self, state, action, target):

        criterion = nn.MSELoss()

        # TODO implement learning rate scheduler with Adam here
        # maybe something like this?     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        optimizer = optim.Adam(self.approx_network.parameters(), lr=0.05) # TODO move outside so we don't need to initialize every time
        optimizer.zero_grad()

        output = self.get_Q_value(state, action, "approx")
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    def monitor_performance(self):
        # get the Q values for the state we want to observe and plot them over time
        state_track = torch.tensor([1, 2, 3, 0], dtype=torch.double)
        best_action = self.get_best_action(state_track, "approx")
        q_best_action = self.get_Q_value(state_track, best_action, "approx")
        print(q_best_action)


    def set_target_weights(self):
        # target network weights and biases <-- approx network weights and biases
        approx_network_state_dict = self.approx_network.state_dict()
        self.target_network.load_state_dict(approx_network_state_dict)


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

def set_target_weights_test():
    env = TestEnv()
    config = LinearConfig
    model = Linear(env, config)

    print("Target weights before: ", model.target_network.fc1.weight)
    print("Approx weights: ", model.approx_network.fc1.weight)
    model.set_target_weights()
    print("======================================================================")
    print("Target weights after: ", model.target_network.fc1.weight)
    print("Approx weights: ", model.approx_network.fc1.weight)

def minibatch_test():
    env = TestEnv()
    config = LinearConfig
    model = Linear(env, config)

    for i in range(1):
        env.reset()
        for state in env.states:
            for action in env.actions:
                if env.done:
                    break
                next_state, reward = env.take_action(state, action)
                experience_tuple = (state, action, reward, next_state, env.done)
                model.replay_buffer.store(experience_tuple)
            if env.done:
                break

    minibatch = model.replay_buffer.sample_minibatch()
    print("Replay buffer: ")
    for exp in model.replay_buffer.replay_buffer:
        print(exp)

    print("==============================================")

    minibatch = model.replay_buffer.sample_minibatch()
    print("Minibatch: ")
    for exp in minibatch:
        print(exp)

def gradient_descent_test():
    # Objective of this test is just to see the weights changing
    # track the Q value of optimal path state for Test Env
    env = TestEnv()
    config = LinearConfig
    model = Linear(env, config)

    # s=[]    a=0    next_s = [1,2,3,1]  done=True
    state_track = torch.tensor([1,2,3,0], dtype=torch.double)
    action_track = torch.tensor(1., dtype=torch.double)

    output = model.get_Q_value(state_track, action_track, "approx")




def main():
    # q_value_test()
    # get_best_action_test()
    # sample_action_test()
    # set_target_weights_test()
    # minibatch_test()
    # gradient_descent_test()
    print("Starting Training")
    env = TestEnv()
    config = LinearConfig
    model = Linear(env, config)
    model.train()
    print("Done")



    pass


if __name__ == '__main__':
    main()

