
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, state_shape, size, env, config):
        self.size = size
        self.config = config
        self.__MAX_SIZE = self.config.replay_buffer_size
        self.__next_replay_location = 0
        self.__state_buffer = np.zeros(shape=(self.__MAX_SIZE, *state_shape))
        print("state buffer shape: ", self.__state_buffer.shape)
        self.__next_index = 0                        # always points to one more than the most recent index of experience
        self.replay_buffer = list()           # stores tuples of experiences where each state in the tuple is a stack of the previous 4 states (including that current state in the stack)
        self.env = env
    # replay buffer handles concatenating last k frames together

    def store(self, experience_tuple):
        # Note: preprocessing of pixels happens within the environment before we get the state.
        # So state already has the pre-processing applied and here we just need to stack the last 4 states we got from each then store that as an experience

        state, action, reward, next_state, done = experience_tuple

        self.__state_buffer[self.__next_index] = state
        self.__next_index += 1
        stacked_state = torch.from_numpy(np.concatenate([self.__state_buffer[ (self.__next_index-1 - i) % self.__MAX_SIZE ] for i in range(self.config.frame_stack_size)]))

        self.__state_buffer[self.__next_index] = next_state
        self.__next_index += 1
        stacked_next_state = torch.from_numpy(np.concatenate([self.__state_buffer[ (self.__next_index-1 - i) % self.__MAX_SIZE ] for i in range(self.config.frame_stack_size)]))

        self.replay_buffer.append((stacked_state, action, reward, stacked_next_state, done))



    def sample_minibatch(self):

        minibatch = []

        for i in range(self.config.minibatch_size):
            minibatch.append( self.replay_buffer[ np.random.randint( len(self.replay_buffer) ) ] )

        return minibatch




















