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

        while self.t <= self.config.nsteps_train:

            # print("Time: ", self.t, " Epsilon: ", epsilon_scheduler.get_epsilon(self.t - self.config.learning_delay), " Learning Rate: ", self.approx_network.optimizer.param_groups[0]['lr'])

            state = self.env.reset()

            while not self.env.done:
                action = self.sample_action(self.env, state, epsilon_scheduler.get_epsilon(self.t - self.config.learning_delay), self.t, "approx")
                next_state, reward = self.env.take_action(state, action)
                experience_tuple = (state, action, reward, next_state, self.env.done)
                self.replay_buffer.store(experience_tuple)
                state = next_state

                if (self.t > self.config.learning_start) and (self.t % self.config.learning_freq == 0):

                    minibatch = self.replay_buffer.sample_minibatch()

                    # turn all the states into torch tensor array
                    # turn all actions in to torch tensor array
                    # compute reward based on done flags and store in torch tensor

                    states, actions, rewards, next_states, dones = map(torch.tensor, *self.replay_buffer.sample_minibatch())
                    states = states.to(dtype=torch.double)
                    rewards = rewards.to(dtype=torch.double)





                    # map, zip, torch.tensor([...]), torch.where

                    # states = []
                    # actions = []
                    # targets = []
                    #
                    # for state_mini, action_mini, reward_mini, next_state_mini, done_mini in minibatch:
                    #
                    #     states.append(state_mini)
                    #     actions.append(action_mini)
                    #
                    #     if done_mini:
                    #         targets.append(torch.tensor(reward_mini, dtype=torch.double))
                    #     else:
                    #         best_action = self.get_best_action(next_state_mini, "target")
                    #         targets.append(reward_mini + (self.config.gamma * self.get_Q_value(next_state_mini, best_action, "target").detach()))   # detach()???
                    #
                    # states = torch.tensor(states, dtype=torch.double)
                    # actions = torch.tensor(actions)
                    # targets = torch.tensor(targets)





                    #     self.perform_gradient_descent(state_mini, action_mini, self.config, target=y, timestep=self.t)







                self.monitor_performance(self.env, timestep=self.t)
                self.t += 1
                if (self.t > self.config.learning_start) and (self.t % self.config.target_weight_update_freq == 0):
                    self.set_target_weights()
