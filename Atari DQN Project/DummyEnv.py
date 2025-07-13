import numpy as np
import torch

class DummyEnv:
    def __init__(self, shape=(5,5,1)):

        self.numStates = 10
        self.numActions = 3
        self.state_shape = shape
        
        self.__state_index = 0
        # self.states = [i for i in range(self.numStates)]

        # make the states have variance in their specific values and give distinction between states
        state_0 = np.random.randint(0, 50, self.state_shape)
        state_1 = np.random.randint(100, 150, self.state_shape)
        state_2 = np.random.randint(200, 250, self.state_shape)
        state_3 = np.random.randint(300, 350, self.state_shape)
        state_4 = np.random.randint(400, 450, self.state_shape)
        state_5 = np.random.randint(500, 550, self.state_shape)
        state_6 = np.random.randint(600, 650, self.state_shape)
        state_7 = np.random.randint(700, 750, self.state_shape)
        state_8 = np.random.randint(800, 850, self.state_shape)
        state_9 = np.random.randint(900, 950, self.state_shape)


        self.states = [state_0, state_1, state_2, state_3, state_4, state_5, state_6, state_7, state_8, state_9] # , state_6, state_7, state_8, state_9

        self.actions = [1, -1, 0]

        self.rewards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] # 0, 0, 0, 0, 1
        self.MAX_TIME_STEPS = 9

        self.current_time = 0
        self.__done = False



    def take_action(self, action):

        assert (self.__done == False)
        assert (0 <= action <= len(self.actions))

        if self.actions[action] == -1 and self.__state_index == 0:
            next_state_index = 0
        else:
            next_state_index = self.__state_index + self.actions[action]

        reward = self.rewards[next_state_index]

        self.current_time += 1
        # if self.current_time >= self.MAX_TIME_STEPS:
        #     self.__done = True

        # for next call to take_action
        self.__state_index = next_state_index


        return torch.tensor(np.array(self.states[next_state_index]), dtype=torch.double), reward, (self.current_time >= self.MAX_TIME_STEPS)



    def reset(self):

        self.__state_index = 0
        self.current_time = 0
        self.__done = False

        return torch.tensor(np.array(self.states[self.__state_index]), dtype=torch.double)

def environmentTest():

    print("State\tAction\tNext State\tReward")

    dummy_env = DummyEnv()
    for s in range(dummy_env.numStates -1): # -1 because last state is terminal state
        for a in range(dummy_env.numActions):
            s = torch.tensor(s, dtype=torch.double)

            next_state, reward = dummy_env.take_action(s, a)

            print(int(s.numpy()), "\t\t", a, "\t\t", int(next_state.numpy()), "\t\t\t", reward)

if __name__ == '__main__':
    # environmentTest()
    pass