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

        self.states = torch.from_numpy(np.array([i for i in range(self.numStates)], dtype=float))

        self.actions = torch.from_numpy(np.array([i for i in range(self.numActions)], dtype=float))
        self.rewards = [0.1, -0.2, 0, -0.1]
        self.MAX_TIME_STEPS = 5

        self.state = self.states[0]
        self.current_time = 0
        self.done = False

    def take_action(self, state, action) -> (int, int):

        assert(self.done == False)
        assert(0 <= state <= len(self.states))
        assert(0 <= action <= len(self.actions))

        next_state = None
        reward = None

        if action == 4:
            next_state = state
        else:
            next_state = action

        reward = self.rewards[int(next_state.numpy())]       # Here I make assumption that we only need to get reward for state based on last state in stack
                                                    # this may cause problems for the test environment I think - but might not be an issue if we visit all s,a infinitely often

        if state == 2:
            reward *= -10

        self.current_time += 1
        if self.current_time >= self.MAX_TIME_STEPS:
            self.done = True

        return next_state, reward


    def reset(self):
        self.state = self.states[0]
        self.current_time = 0
        self.done = False

        return self.state


def environmentTest():

    print("State\tAction\tNext State\tReward")

    test_env = TestEnv()
    for s in test_env.states:
        for a in test_env.actions:

            next_state, reward = test_env.take_action(s, a)

            print(s, "\t\t", a, "\t\t", next_state, "\t\t\t", reward)



if __name__ == '__main__':
    environmentTest()
    pass