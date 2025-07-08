
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
        # self.replay_buffer = [None for i in range(self.__MAX_SIZE)]           # stores tuples of experiences where each state in the tuple is a stack of the previous 4 states (including that current state in the stack)

        self.replay_buffer = set()               # just to see if the buffer contains many of the same experiences

        self.env = env

    def store(self, experience_tuple):
        # self.replay_buffer[self.__next_replay_location] = experience_tuple

        # TODO: CHANGE SO WE USE THIS AS A SET INSTEAD - will have to change store because we want to return the state and next_state as tuples
        # BUT WE WANT TO STORE THEM AS .ITEM() BC SET WILL HASH ON THAT
        # TODO
        # - change sample_minibatch() function to support ^^^ and

        # THIS
        state = experience_tuple[0].item()
        action = experience_tuple[1]
        reward = experience_tuple[2]
        next_state = experience_tuple[3].item()
        done = experience_tuple[4]


        self.replay_buffer.add((state, action, reward, next_state, done))



        # self.__next_replay_location = (self.__next_replay_location + 1) % self.__MAX_SIZE
        # self.__num_elements  = min(self.__num_elements + 1, self.__MAX_SIZE)


    def sample_minibatch(self):

        # TODO
        '''
        want to sample a minibatch then I want to make minibatch = states, action, rewards, next_states, dones
        where each element of that is a list of corresponding values of shape [batch_size]
        '''

        minibatch = []

        buffer = list(self.replay_buffer)

        for i in range(self.config.minibatch_size):
            minibatch.append( buffer[ np.random.randint( len(self.replay_buffer) ) ] )

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for i in range(len(minibatch)):
            states.append( torch.tensor([minibatch[i][0]], dtype=torch.double)   )  # must be torch
            actions.append(int(minibatch[i][1]))                                            # int
            rewards.append(minibatch[i][2])                                                 # double/float I forget
            next_states.append(torch.tensor([minibatch[i][3]], dtype=torch.double))                                             # must be torch
            dones.append(minibatch[i][4])


        return (
            torch.tensor(states, dtype=torch.double),       # states contains tensors
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.double),
            torch.tensor(next_states, dtype=torch.double),  # next_states contains tensors
            torch.tensor(dones, dtype=torch.bool)
        )














