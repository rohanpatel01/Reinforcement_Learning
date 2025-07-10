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

        self.states = [i for i in range(self.numStates)]

        # Represent state using Random Floats within intervals
        # Upon returning them from take_action() and reset() we will normalize them
        # self.states = []
        # state_0 = np.random.randint(0, 100)
        # state_0 = np.random.randint(0, 100)
        # state_0 = np.random.randint(0, 100)
        # state_0 = np.random.randint(0, 100)
        # state_0 = np.random.randint(0, 100)

        self.actions = [i for i in range(self.numActions)]

        self.rewards = [0.1, -0.2, 0, -0.1]
        self.MAX_TIME_STEPS = 5

        self.state = self.states[0]
        self.current_time = 0
        self.done = False

    def take_action(self, state, action) -> (int, int):

        assert(self.done == False)
        assert(type(state) == torch.Tensor)
        assert(type(action) == int)
        assert(0 <= action <= len(self.actions))

        state = int(state.numpy())
        assert(0 <= state <= len(self.states))

        if action == 4:
            next_state = state
        else:
            next_state = action

        reward = self.rewards[next_state]

        if state == 2:
            reward *= -10

        self.current_time += 1
        if self.current_time >= self.MAX_TIME_STEPS:
            self.done = True

        return torch.tensor([next_state], dtype=torch.double), reward



    def reset(self):
        self.state = torch.tensor([0], dtype=torch.double)
        self.current_time = 0
        self.done = False

        return self.state


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