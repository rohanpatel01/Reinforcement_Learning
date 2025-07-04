import numpy as np
import torch

class SimpleEnv:
    def __init__(self):

        self.numStates = 3
        self.numActions = 2
        self.state_shape = np.array([1])

        self.states = [i for i in range(self.numStates)]
        self.actions = [1, 2]        # action value is next state

        self.rewards = [0, 0, 1]     # optimal policy is to always go right: i.e always choose action 1: (1)
        self.MAX_TIME_STEPS = 1

        self.current_time = 0
        self.done = False


    def take_action(self, state, action) -> (int, int):
        assert (self.done == False)
        assert (type(state) == torch.Tensor)
        assert (type(action) == int)
        assert (0 <= action <= len(self.actions))

        state = int(state.numpy())
        assert (0 <= state <= len(self.states))

        # Compute next state
        if self.actions[action] == 1:
            next_state = 1
        elif self.actions[action] == 2:
            next_state = 2

        # Compute reward
        reward = self.rewards[next_state]

        # Compute done
        self.current_time += 1
        if self.current_time >= self.MAX_TIME_STEPS:
            self.done = True

        return torch.tensor([next_state], dtype=torch.double), reward



    def reset(self):
        self.current_time = 0
        self.done = False

        return torch.tensor([0], dtype=torch.double)

if __name__ == '__main__':
    pass