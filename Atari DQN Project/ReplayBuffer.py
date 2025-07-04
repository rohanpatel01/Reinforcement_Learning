
import numpy as np

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

        minibatch = []

        for i in range(self.config.minibatch_size):
            minibatch.append( self.replay_buffer[ np.random.randint( self.__num_elements ) ] )

        return minibatch




















