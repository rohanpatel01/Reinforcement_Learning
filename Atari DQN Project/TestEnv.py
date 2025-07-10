import numpy as np
import torch

# Static Methods
def get_initial_state():
    return 0

class TestEnv:
    def __init__(self):

        self.numStates = 4
        self.numActions = 5
        self.state_shape = np.array([1])

        # self.states = [i for i in range(self.numStates)]
        self.__state_index = 0

        # Represent state using Random Floats within intervals
        # Upon returning them from take_action() and reset() we will normalize them
        # self.states = []
        state_0 = np.random.randint(0, 50)
        state_1 = np.random.randint(100, 150)
        state_2 = np.random.randint(200, 250)
        state_3 = np.random.randint(300, 350)

        self.states = [state_0, state_1, state_2, state_3]

        self.actions = [i for i in range(self.numActions)]

        self.rewards = [0.1, -0.2, 0, -0.1]
        self.MAX_TIME_STEPS = 5

        # self.state = self.states[0]
        self.current_time = 0
        self.__done = False

    def take_action(self, action) -> (int, int):

        assert(self.__done == False)
        assert(0 <= action <= len(self.actions))

        if action == 4:
            next_state_index = self.__state_index
        else:
            next_state_index = action

        reward = self.rewards[next_state_index]

        if self.__state_index == 2:
            reward *= -10

        self.current_time += 1
        # if self.current_time >= self.MAX_TIME_STEPS:
        #     self.done = True
        self.__state_index = next_state_index

        return torch.tensor([self.states[next_state_index]], dtype=torch.double), reward, self.current_time >= self.MAX_TIME_STEPS



    def reset(self):
        self.__state_index = 0
        # self.state = torch.tensor([0], dtype=torch.double)
        self.current_time = 0
        self.__done = False

        return torch.tensor([self.states[self.__state_index]], dtype=torch.double)


def environmentTest():

    print("State\tAction\tNext State\tReward")

    test_env = TestEnv()
    for s in test_env.states:
        for a in test_env.actions:
            s = torch.tensor(s, dtype=torch.double)

            next_state, reward = test_env.take_action(s, a)

            print(int(s.numpy()), "\t\t", a, "\t\t", int(next_state.numpy()), "\t\t\t", reward)



if __name__ == '__main__':
    # environmentTest()
    pass