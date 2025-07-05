import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import time
from Q_Learning import Q_Learning
from TestEnv import TestEnv
from DummyEnv import DummyEnv
from SimpleEnv import SimpleEnv
from Config.LinearConfig import LinearConfig
from Scheduler import EpsilonScheduler

writer = SummaryWriter()

class Linear(Q_Learning):

    def __init__(self, env, config):
        super(Linear, self).__init__(env, config)
        self.env = env
        self.config = config
        self.target_network = LinearNN(env, config)
        self.approx_network = LinearNN(env, config)
        self.set_target_weights()                   # Initially we want both target and approx network to have the same arbitrary weights


    def sample_action(self, env, state, epsilon, network_name):

        if np.random.rand() < epsilon:
            # take random action
            return np.random.randint(env.numActions)
        else:
            # select greedy action
            return self.get_best_action(state, network_name)

    def get_Q_value(self, state, action, network_name):
        # Note: action is the index of the action defined in the environment's action space
        assert(network_name == "target" or network_name == "approx")

        if network_name == "target":
            Q_actions = self.target_network.forward(state)

        elif network_name == "approx":
            Q_actions = self.approx_network.forward(state)

        # action here is an int and not a tensor so we don't need to convert it
        return Q_actions[action] # int(action.numpy())

    # Returns index of action in action space with highest Q value
    def get_best_action(self, state, network_name) -> int:
        if network_name == "target":
            Q_actions = self.target_network.forward(state)
        elif network_name == "approx":
            Q_actions = self.approx_network.forward(state)

        best_action_index = torch.argmax(Q_actions.detach())
        return best_action_index.item()
        # return int(best_action_index.numpy())

    def perform_gradient_descent(self, state, action, target, timestep):
        # TODO implement learning rate scheduler with Adam here
        # maybe something like this?     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        output = self.get_Q_value(state, action, "approx")
        loss = self.approx_network.criterion(output, target)
        # print(f"Logging loss: {loss.item()} at timestep {timestep}")
        writer.add_scalar("Loss/train", loss.item(), timestep)
        self.approx_network.optimizer.zero_grad()
        loss.backward()

        # print("Gradients for fc1 layer:")
        # for name, param in self.approx_network.fc1.named_parameters():
        #     if param.grad is not None:
        #         print(f"  Parameter: {name}")
        #         print(f"    Gradient shape: {param.grad.shape}")
        #         print(f"    Gradient values (first 5): {param.grad.flatten()[:5]}")
        #         print(f"    Gradient mean: {param.grad.mean().item():.6f}")
        #         print("-" * 20)
        #     else:
        #         print(f"  Parameter: {name} has no gradient (check if requires_grad=True or if backward was called)")
        #         print("-" * 20)

        self.approx_network.optimizer.step()

    def post_minibatch_updates(self):
        # self.approx_network.scheduler.step()
        pass

    def monitor_performance(self, env, timestep):
        # state_track = torch.tensor([0], dtype=torch.double)
        # # action = self.get_best_action(state_track, "approx")
        # q_best_action = self.get_Q_value(state_track, action=1, network_name="approx")
        # # q_best_action = self.get_Q_value(state_track, action, "approx")
        # # print("Q of state: ", state_track, " and best action ", best_action, " : ", q_best_action)
        # writer.add_scalar("Performance/TestEnv_State_0_Action_2", q_best_action, timestep)

        # Monitor average Q(s,a) over time
        count = 0
        sum = 0
        for state in env.states:
            for action in range(env.numActions):
                state = torch.tensor([state], dtype=torch.double).detach()
                sum += self.get_Q_value(state, action, "approx").detach()
                count += 1
        # Trying to check for divergence. Divergence would appear as monotonic increase in Q-values even after the policy stops improving
        writer.add_scalar("Performance/DummyEnv_Average_Q(s,a)", sum/count, timestep)



    def set_target_weights(self):
        # target network weights and biases <-- approx network weights and biases
        approx_network_state_dict = self.approx_network.state_dict()
        self.target_network.load_state_dict(approx_network_state_dict)


class LinearNN(nn.Module):
    def __init__(self, env, config):
        # torch.manual_seed(42)
        super(LinearNN, self).__init__()
        self.env = env
        self.config = config
        self.fc1 = nn.Linear(np.prod(self.env.state_shape)*self.config.frame_stack_size, self.env.numActions)
        self.double()       # converts model to double to match datatype of input with no serious performance problems
        self.optimizer = optim.Adam(self.parameters(), lr=self.config.lr_begin)

        # gamma = (self.config.lr_end / self.config.lr_begin) ** (self.config.nsteps_train / self.config.step_size)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config.step_size, gamma=gamma)
        self.criterion = nn.MSELoss()
        print("layer weights: ", self.fc1.weight)

    def forward(self, x):
        x = self.fc1(x)
        return x


def summary(model, env, config):

    state = env.reset()

    print()
    print("==============================================")
    print("Summary: ")
    # Print out all Q(s,a) values
    print("State\tAction\tNext State\tReward")

    for state in env.states:
        for action in range(len(env.actions)):
            state = torch.tensor([state], dtype=torch.double)
            print(int(state[0].numpy()), "\t\t", action, "\t\t", " Q(s,a)= ", model.get_Q_value(state, action, "approx"))

    print("==============================================")
    print()
    print("Best trajectory: ")

    # Get best trajectory from state 0
    state = env.reset()

    states_visited = []
    actions_taken = []
    rewards_received = []

    while not env.done:
        best_action = model.get_best_action(state, "approx")

        states_visited.append(state)
        actions_taken.append(best_action)

        next_state, reward = env.take_action(state, best_action)
        rewards_received.append(reward)

        state = next_state

    print("Best trajectory from Test Environment")
    for i in range(len(states_visited)):
        s = states_visited[i]
        a = actions_taken[i]
        r = rewards_received[i]

        print("State: ", s, " Action: ", a, " Reward Received: ", r)

    print()
    print("Total Reward Received: ", sum(rewards_received))

    print("Taking a look at model parameters to see if weights are changing")
    print(model.approx_network.fc1.weight)
    print(model.approx_network.fc1.bias)

    print("Configs:")
    print("Num episodes: ", config.num_episodes)
    print("Num nsteps_train %: ", config.nsteps_train / config.num_episodes)
    print()




def q_value_test():
    env = TestEnv()
    config = LinearConfig
    model = Linear(env, config)

    for state in env.states:
        for action in env.actions:
            if env.done:
                break
            next_state, reward = env.take_action(state, action)
            experience_tuple = (state, action, reward, next_state, env.done)
            model.replay_buffer.store(experience_tuple)
        if env.done:
            break

    state, action, reward, next_state, done = model.replay_buffer.replay_buffer[0]
    Q_state_action = model.get_Q_value(state, action, "target")
    print("Q value for state: ", state, " and action: ", action, " : ", Q_state_action)

def get_best_action_test():
    env = TestEnv()
    config = LinearConfig
    model = Linear(env, config)

    for state in env.states:
        for action in env.actions:
            if env.done:
                break
            next_state, reward = env.take_action(state, action)
            experience_tuple = (state, action, reward, next_state, env.done)
            model.replay_buffer.store(experience_tuple)
        if env.done:
            break

    state, action, reward, next_state, done = model.replay_buffer.replay_buffer[0]
    # Show Q action values of state
    best_action = model.get_best_action(state, "target")
    print("best action: ", best_action)

    # Show best action

def sample_action_test():

    # torch.manual_seed(42)

    env = TestEnv()
    config = LinearConfig
    model = Linear(env, config)


    for state in env.states:
        for action in env.actions:
            if env.done:
                break
            next_state, reward = env.take_action(state, action)
            experience_tuple = (state, action, reward, next_state, env.done)
            model.replay_buffer.store(experience_tuple)
        if env.done:
            break

    state, action, reward, next_state, done = model.replay_buffer.replay_buffer[0]

    epsilon_scheduler = EpsilonScheduler(config.begin_epsilon, config.end_epsilon,
                                         config.max_time_steps_update_epsilon)

    e = .4

    total_runs = 1000000
    total_greedy = 0
    total_random = 0

    best_action = model.get_best_action(state, "target")

    print("State: ", state)
    print("Best action: ", best_action)
    print(model.target_network.forward(state))

    for i in range(total_runs):
        action = model.sample_action(env, state, e, "target")

        if action == best_action:
            total_greedy += 1
        else:
            total_random += 1

    print("Epsilon (% take random action): ", e)
    print("Greedy: ", total_greedy / total_runs, "%")
    print("Random: ", total_random / total_runs, "%")

def set_target_weights_test():
    env = TestEnv()
    config = LinearConfig
    model = Linear(env, config)

    print("Target weights before: ", model.target_network.fc1.weight)
    print("Approx weights: ", model.approx_network.fc1.weight)
    model.set_target_weights()
    print("======================================================================")
    print("Target weights after: ", model.target_network.fc1.weight)
    print("Approx weights: ", model.approx_network.fc1.weight)

def minibatch_test():
    env = TestEnv()
    config = LinearConfig
    model = Linear(env, config)

    for i in range(1):
        env.reset()
        for state in env.states:
            for action in env.actions:
                if env.done:
                    break
                next_state, reward = env.take_action(state, action)
                experience_tuple = (state, action, reward, next_state, env.done)
                model.replay_buffer.store(experience_tuple)
            if env.done:
                break

    minibatch = model.replay_buffer.sample_minibatch()
    print("Replay buffer: ")
    for exp in model.replay_buffer.replay_buffer:
        print(exp)

    print("==============================================")

    minibatch = model.replay_buffer.sample_minibatch()
    print("Minibatch: ")
    for exp in minibatch:
        print(exp)

def gradient_descent_test():
    # Objective of this test is just to see the weights changing
    # track the Q value of optimal path state for Test Env
    # env = TestEnv()
    # config = LinearConfig
    # model = Linear(env, config)

    # s=[]    a=0    next_s = [1,2,3,1]  done=True
    # state_track = torch.tensor([1,2,3,0], dtype=torch.double)
    # action_track = torch.tensor(1., dtype=torch.double)

    # output = model.get_Q_value(state_track, action_track, "approx")
    pass



def main():
    # q_value_test()
    # get_best_action_test()
    # sample_action_test()
    # set_target_weights_test()
    # minibatch_test()
    # gradient_descent_test()
    print("Starting Training")
    # env = DummyEnv()
    # env = SimpleEnv()

    # need to perform some hyperparameter search
    env = TestEnv()
    config = LinearConfig

    # best_reward = float('-inf')
    # best_n_episodes = -1
    # best_n_steps = -1

    # orig = config.nsteps_train
    # for n_episodes in config.num_episodes:
    #     for nsteps in config.nsteps_train:
    #
    #         config.num_episodes = n_episodes
    #         config.nsteps_train = config.num_episodes * nsteps
    #         config.max_time_steps_update_epsilon = config.nsteps_train

    for i in range(10):
        model = Linear(env, config)
        model.train()
        writer.flush()
        writer.close()

        print("Summary AFTER training")
        summary(model, env, config)

        #     if total_reward > best_reward:
        #         best_reward = total_reward
        #         best_n_episodes = n_episodes
        #         best_n_steps = (config.nsteps_train / config.num_episodes)
        #
        # config.nsteps_train = orig

    # print("Best Reward: ", best_reward)
    # print("Best n_episodes: ", best_n_episodes)
    # print("Best n_steps: ", best_n_steps)



if __name__ == '__main__':


    main()


