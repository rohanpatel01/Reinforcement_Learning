import numpy as np

# This will be a general class to implement Q_learning with the following params to make it flexible
'''
learning rate scheduler
epsilon scheduler
state space
action space
policy
reward space

have stuff to monitor:
    - average reward
    - Average Q (Average the max predicted Q of the fixed set of states we choose to observe)

'''

import ReplayBuffer
from Scheduler import EpsilonScheduler

class Q_learning(object):
    # This is a general class that creates a framework for Q learning with experience replay

    def __init__(self, epsilon, env, config):
        self.env = env
        self.config = config
        self.replay_buffer = ReplayBuffer(self.config.replay_buffer_size)
        self.time = 0

    def build(self):
        # builds and initializes the NN that will be used for function approximation for Q values
        pass

    def get_initial_state(self, env):
        raise NotImplementedError

    def sample_action(self, env, state, epsilon):
        # sample an action with e-greedy policy given by input parameter
        raise NotImplementedError

    def get_Q_value(self, preprocessed_state, action, network_name):
        raise NotImplementedError

    def get_best_action(self, preprocessed_state, network_name) -> int:
        # TODO: for the given preprocessed_state, we want to return the index of the action in action space that gives the highest Q value
        pass

    def preprocess(self, state):
        # state here is the history of states up till this point
        # state = (previous state, previous action, reward, current pixels)
        # we want to iterate backwards through this representation of history (state) and from the last 4 raw pixels:
        #   preprocess each of the raw pixels
        #   stack them to produce (84x84x4) shape


        # return the preprocessing of input state
        raise NotImplementedError


    # TODO: May need some input parameters such as loss or something - figure this out when the time comes
    def perform_gradient_descent(self):
        # TODO: figure out if we need autograd here and how it works / what we need to do here to get it working
        # in order to update weights based on the loss
        raise NotImplementedError

    def train(self):

        for episode in range(self.config.num_episodes):

            state = self.get_initial_state(self, self.env)
            epsilon_scheduler = EpsilonScheduler(self.config.begin_epsilon, self.config.end_epsilon,
                                                 self.config.max_time_steps_update_epsilon)

            state_sequence = [(state)]
            preprocessed_sequence = [(self.preprocess(state))]

            while not self.env.done:

                action = self.sample_action(self.env, state, epsilon_scheduler.get_epsilon(self.time))
                next_state, reward = self.env.take_action(state, action)

                state_sequence.append((state_sequence[-1], action, next_state))
                preprocessed_sequence.append(self.preprocess(state_sequence[-1]))

                self.replay_buffer.store((preprocessed_sequence[-2], action, reward, preprocessed_sequence[-1], self.env.done))
                minibatch = self.replay_buffer.sample_minibatch()

                for preprocessed_state, action, reward, preprocessed_next_state, done in minibatch:
                    if done:
                        y = reward
                    else:
                        best_action = self.get_best_action(preprocessed_next_state, 'target')
                        y = reward + (self.env.gamma * self.get_Q_value(self, preprocessed_next_state, best_action, 'target'))

                    # perform gradient updates
                    # TODO: Figure out this part
                    self.perform_gradient_descent()

                # perform updates for end of time step
                self.time += 1
                # if (self.time % self.config.learning_freq) == 0:
                # update target_weights with current weights
                epsilon_scheduler.update_epsilon(self.time)
                # done with current time step

            # done with episode
            self.env.reset()

























