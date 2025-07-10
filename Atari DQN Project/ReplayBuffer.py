
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, state_shape, size, env, config):
        self.size = size
        self.config = config
        self.__MAX_SIZE = self.config.replay_buffer_size
        self.__next_replay_location = 0
        # self.__state_buffer = np.zeros(shape=(self.__MAX_SIZE, *state_shape))
        # print("state buffer shape: ", self.__state_buffer.shape)
        # self.__next_index = 0                        # always points to one more than the most recent index of experience
        self.__experience_shape = [5] # because we just store the one element: the tuple
        self.__num_elements = 0
        self.replay_buffer = [None for i in range(self.__MAX_SIZE)]           # stores tuples of experiences where each state in the tuple is a stack of the previous 4 states (including that current state in the stack)
        self.env = env

    def store(self, experience_tuple):
        self.replay_buffer[self.__next_replay_location] = experience_tuple
        self.__next_replay_location = (self.__next_replay_location + 1) % self.__MAX_SIZE
        self.__num_elements  = min(self.__num_elements + 1, self.__MAX_SIZE)


    def sample_minibatch(self):

        # TODO
        '''
        want to sample a minibatch then I want to make minibatch = states, action, rewards, next_states, dones
        where each element of that is a list of corresponding values of shape [batch_size]
        '''

        minibatch = []

        for i in range(self.config.minibatch_size):
            minibatch.append( self.replay_buffer[ np.random.randint( self.__num_elements ) ] )

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

        # return (
        #     torch.tensor(states, dtype=torch.double),
        #     torch.tensor(actions, dtype=torch.long),
        #     torch.tensor(rewards, dtype=torch.double),
        #     torch.tensor(next_states, dtype=torch.double),
        #     torch.tensor(dones, dtype=torch.bool)
        # )

        return (
            torch.tensor(np.array(states)),
            torch.tensor(np.array(actions)),
            torch.tensor(np.array(rewards)),
            torch.tensor(np.array(next_states)),
            torch.tensor(np.array(dones)),
        )



















