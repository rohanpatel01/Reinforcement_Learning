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
import time



class Q_Learning:
    # This is a general class that creates a framework for Q learning with experience replay

    def __init__(self, env, config, device):
        self.env = env
        self.config = config
        self.device = device
        self.replay_buffer = ReplayBuffer(self.config.replay_buffer_size, self.env, self.config)
        self.t = 1
        self.num_episodes = 0
        self.total_reward_so_far = 0

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

    def process_state(self, state):
        # state = state.double()
        state = state.float()
        state /= self.config.high
        return state

    def save_snapshop(self):
        pass

    def load_snapshot(self):
        pass


    def train(self):

        epsilon_scheduler = EpsilonScheduler(self.config.begin_epsilon, self.config.end_epsilon,
                                             self.config.max_time_steps_update_epsilon)

        time_last_saved = self.t

        while self.t <= self.config.nsteps_train:

            print("Time: ", self.t, " Epsilon: ", epsilon_scheduler.get_epsilon(self.t - self.config.learning_delay), " Learning Rate: ", self.approx_network.optimizer.param_groups[0]['lr'])

            if (self.t - time_last_saved) >= self.config.saving_freq:
                self.save_snapshop(self.t, self.num_episodes, self.total_reward_so_far, self.replay_buffer) # just for development now we're not gonna save snapshots
                time_last_saved = self.t

            state, info = self.env.reset()
            state = torch.from_numpy(state)
            state = self.process_state(state)
            state = state.to(self.device)

            total_reward_for_episode = 0
            start_time = time.time()

            while True:
                with torch.no_grad():

                    # TODO: Note: when we sample action - in the get_best_action function we can also monitor the highest Q values - can help us see if we're training right
                    action = self.sample_action(self.env, state, epsilon_scheduler.get_epsilon(self.t - self.config.learning_delay), self.t, "approx")

                    next_state, reward, terminated, truncated, info = self.env.step(action)
                    next_state = torch.from_numpy(next_state)
                    next_state = self.process_state(next_state).to(self.device)

                    # convert state and next_state to uint8 before placing in replay buffer
                    experience_tuple = ((state*self.config.high).to(torch.uint8).to('cpu'), action, reward, (next_state*self.config.high).to(torch.uint8).to('cpu'), terminated)
                    self.replay_buffer.store(experience_tuple)

                state = next_state

                if (self.t > self.config.learning_start) and (self.t % self.config.learning_freq == 0):
                    self.train_on_minibatch(self.replay_buffer.sample_minibatch(), self.t)

                # Measures Max_Q per timestep and evaluates agent when time comes
                self.monitor_performance(state, reward, monitor_end_of_episode=False, timestep=self.t)    # used to have env as param
                # torch.cuda.empty_cache()
                self.total_reward_so_far += reward
                total_reward_for_episode += reward

                self.t += 1
                if (self.t > self.config.learning_start) and (self.t % self.config.target_weight_update_freq == 0):
                    self.set_target_weights()

                if terminated:
                    # monitor avg reward per episode and max_reward per episode (at end of episode)
                    self.num_episodes += 1
                    self.monitor_performance(state, reward, monitor_end_of_episode=True, timestep=self.t, context = (self.total_reward_so_far, self.num_episodes, total_reward_for_episode))
                    break

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time for episode: {elapsed_time:.6f} seconds", " : Device: ", self.device)
