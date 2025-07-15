
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, size, env, config):
        self.size = size
        self.config = config
        self.__MAX_SIZE = self.config.replay_buffer_size
        self.next_replay_location = 0
        self.num_elements = 0
        self.replay_buffer = np.array([None for i in range(self.__MAX_SIZE)])           # stores tuples of experiences where each state in the tuple is a stack of the previous 4 states (including that current state in the stack)
        self.env = env

    def store(self, experience_tuple):
        self.replay_buffer[self.next_replay_location] = experience_tuple
        self.next_replay_location = (self.next_replay_location + 1) % self.__MAX_SIZE
        self.num_elements  = min(self.num_elements + 1, self.__MAX_SIZE)


    def sample_minibatch(self):

        minibatch = []

        for i in range(self.config.minibatch_size):
            minibatch.append( self.replay_buffer[ np.random.randint( self.num_elements ) ] )

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for i in range(len(minibatch)):
            states.append(minibatch[i][0])
            actions.append(int(minibatch[i][1]))
            rewards.append(minibatch[i][2])
            next_states.append(minibatch[i][3])
            dones.append(minibatch[i][4])

        return (
            torch.stack(states),
            torch.tensor(actions),
            torch.tensor(rewards),
            torch.stack(next_states),
            torch.tensor(dones),
        )



















