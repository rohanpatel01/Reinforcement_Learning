
import numpy as np

class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.__MAX_SIZE = 1e6
        self.__next_replay_location = 0
        self.__replay_buffer = list()           # stores tuples of experiences that are preprocessed

    def store(self, experience_tuple):

        """
        TODO
            Note: Experience_tuple is: preprocessed state t, At, Rt, preprocessed next state T+1

            make sure to place new tuple in correct location - accounting for replacement of newest
            then need to store in D as (preprocessed t, At, Rt, next preprocessed t)

        """
        raise NotImplementedError


    def sample_minibatch(self):
        """
        TODO
            need to sample random tuples from

        """
        raise NotImplementedError