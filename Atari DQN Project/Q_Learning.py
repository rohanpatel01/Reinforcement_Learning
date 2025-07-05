import numpy as np
import torch
from tqdm import tqdm

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
        self.time = 0

    def sample_action(self, env, state, epsilon, network_name):
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

    def build(self):
        # builds and initializes the NN that will be used for function approximation for Q values
        # I think this can be optional
        pass

    def monitor_performance(self):
        pass

    def post_minibatch_updates(self):
        pass

    def train(self):

        epsilon_scheduler = EpsilonScheduler(self.config.begin_epsilon, self.config.end_epsilon,
                                             self.config.max_time_steps_update_epsilon)

        for episode in range(self.config.num_episodes):
            print("Episode: ", episode + 1, "Time: ", self.time, " Epsilon: ", epsilon_scheduler.get_epsilon(self.time))

            state = self.env.reset()

            while not self.env.done:
                action = self.sample_action(self.env, state, epsilon_scheduler.get_epsilon(self.time), "approx")
                next_state, reward = self.env.take_action(state, action)
                experience_tuple = (state, action, reward, next_state, self.env.done)
                self.replay_buffer.store(experience_tuple)
                state = next_state

                if (self.time > self.config.learning_start) and (self.time % self.config.learning_freq == 0):

                    minibatch = self.replay_buffer.sample_minibatch()

                    for state_mini, action_mini, reward_mini, next_state_mini, done_mini in minibatch:
                        if done_mini:
                            y = torch.tensor(reward_mini, dtype=torch.double)
                        else:
                            best_action = self.get_best_action(next_state_mini, "target")
                            # Important note: .detach below is necessary to prevent pytorch from doing backprop through the target network
                            y = reward_mini + (self.config.gamma * self.get_Q_value(next_state_mini, best_action, "target").detach())

                        self.perform_gradient_descent(state_mini, action_mini, target=y, timestep=self.time)

                self.post_minibatch_updates()
                self.monitor_performance(self.env, timestep=self.time)
                self.time += 1
                # TODO: here is where we would update the learning rate
                # Something like: scheduler.step() where scheduler = scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
                if (self.time > self.config.learning_start) and (self.time % self.config.target_weight_update_freq == 0):
                    self.set_target_weights()

                # done with current time step
            # done with episode

