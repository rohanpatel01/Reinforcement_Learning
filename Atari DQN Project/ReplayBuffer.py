
import numpy as np

class ReplayBuffer:
    def __init__(self, state_shape, size, env, config):
        self.size = size
        self.config = config
        # self.__MAX_SIZE = self.config.replay_buffer_size
        # self.__next_replay_location = 0
        # self.__state_buffer = np.zeros(shape=(self.__MAX_SIZE, *state_shape))
        # print("state buffer shape: ", self.__state_buffer.shape)
        # self.__next_index = 0                        # always points to one more than the most recent index of experience
        self.replay_buffer = list()           # stores tuples of experiences where each state in the tuple is a stack of the previous 4 states (including that current state in the stack)
        self.env = env

    def store(self, experience_tuple):
        self.replay_buffer.append(experience_tuple)

    def sample_minibatch(self):

        minibatch = []

        for i in range(self.config.minibatch_size):
            minibatch.append( self.replay_buffer[ np.random.randint( len(self.replay_buffer) ) ] )

        return minibatch




















