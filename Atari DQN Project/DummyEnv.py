import numpy as np
import torch

class DummyEnv:
    def __init__(self):

        self.numStates = 10 # 3
        self.numActions = 3
        self.state_shape = np.array([1])

        self.states = [i for i in range(self.numStates)]
        self.actions = [1, -1, 0]      # Action 0: -1         Action 1: 1

        self.rewards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        # self.rewards = [0, 0, 1]
        self.MAX_TIME_STEPS = 9 # 2

        self.current_time = 0
        self.done = False



    def take_action(self, state, action) -> (int, int):
        assert (self.done == False)
        assert (type(state) == torch.Tensor)
        assert (type(action) == int)
        assert (0 <= action <= len(self.actions))

        state = int(state.numpy())
        assert (0 <= state <= len(self.states))

        if self.actions[action] == -1 and state == 0:
            next_state = 0
        else:
            next_state = state + self.actions[action]

        reward = self.rewards[next_state]

        self.current_time += 1
        if self.current_time >= self.MAX_TIME_STEPS:
            self.done = True

        return torch.tensor([next_state], dtype=torch.double), reward



    def reset(self):
        self.current_time = 0
        self.done = False

        return torch.tensor([0], dtype=torch.double)

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