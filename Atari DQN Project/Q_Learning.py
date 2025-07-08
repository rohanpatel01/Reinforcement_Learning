import numpy as np
import torch
from tqdm import tqdm
from collections import Counter

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

from ReplayBuffer import ReplayBuffer
from Scheduler import EpsilonScheduler


class Q_Learning:
    # This is a general class that creates a framework for Q learning with experience replay

    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.replay_buffer = ReplayBuffer(self.env.state_shape, self.config.replay_buffer_size, self.env, self.config)
        self.t = 1

    def sample_action(self, env, state, epsilon, time, network_name):
        # sample an action with e-greedy policy given by input parameter
        raise NotImplementedError

    def get_Q_value(self, state, action, network_name):
        raise NotImplementedError

    def get_best_action(self, state, network_name) -> int:
        # TODO: for the given state, we want to return the index of the action in action space that gives the highest Q value
        raise NotImplementedError

    # TODO: May need some input parameters such as loss or something - figure this out when the time comes
    def perform_gradient_descent(self, state, action, target):
        # TODO: figure out if we need autograd here and how it works / what we need to do here to get it working
        # in order to update weights based on the loss
        pass

    def set_target_weights(self):
        # replace target weights with that of the current network's weights
        raise NotImplementedError

    def train_on_minibatch(self, minibatch, timestep):
        pass

    def monitor_performance(self):
        pass

    def evaluate_policy(self):
        pass

    def train(self):

        epsilon_scheduler = EpsilonScheduler(self.config.begin_epsilon, self.config.end_epsilon,
                                             self.config.max_time_steps_update_epsilon)

        while self.t <= self.config.nsteps_train:
            self.t += 1

            print("Time: ", self.t, " Epsilon: ", epsilon_scheduler.get_epsilon(self.t - self.config.learning_delay), " Learning Rate: ", self.approx_network.optimizer.param_groups[0]['lr'])

            state = self.env.reset()

            while True:         # not self.env.done
                action = self.sample_action(self.env, state, epsilon_scheduler.get_epsilon(self.t - self.config.learning_delay), self.t, "approx")
                next_state, reward = self.env.take_action(state, action)
                experience_tuple = (state, action, reward, next_state, self.env.done)
                self.replay_buffer.store(experience_tuple)
                state = next_state

                if (self.t > self.config.learning_start) and (self.t % self.config.learning_freq == 0):
                    self.train_on_minibatch(self.replay_buffer.sample_minibatch(), self.t)

                self.monitor_performance(reward, self.env, timestep=self.t)

                # Perform update if it is time
                if (self.t > self.config.learning_start) and (self.t % self.config.target_weight_update_freq == 0):
                    self.set_target_weights()


                if self.env.done:
                    break


            # just for logging purposes we will count the number of unique elements in replay buffer along with count per unique element
            # just doing this less often
            # if self.t % 500 == 0:
            #     print()
            #     print("=====================================================================")
            #     print("Time: ", self.t)
            #
            #     counts = Counter(self.replay_buffer.temp_buffer)
            #     num_unique = len(counts)
            #
            #     print("Number of unique items:", num_unique)
            #     print("Counts per unique item:")
            #     for item, count in counts.items():
            #         print(f"{item}: {count}")
            #
            #     print("=====================================================================")
            #     print()


            # Evaluate the current policy
            if self.t % 500 == 0:
                self.evaluate_policy()


















