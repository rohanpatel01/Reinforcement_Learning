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
import optuna
from optuna_dashboard import run_server

writer = SummaryWriter()

class Linear(Q_Learning):

    def __init__(self, env, config):
        super(Linear, self).__init__(env, config)
        self.env = env
        self.config = config
        self.target_network = LinearNN(env, config)
        self.approx_network = LinearNN(env, config)
        self.set_target_weights()                   # Initially we want both target and approx network to have the same arbitrary weights


    def sample_action(self, env, state, epsilon, time, network_name):

        # state = torch.flatten(state)    # using default start_dim=0 here because

        if  (time < self.config.learning_delay )or  (np.random.rand() < epsilon):
            # take random action
            return np.random.randint(env.numActions)
        else:
            # select greedy action
            return self.get_best_action(state, network_name)

    def get_Q_value(self, state, action, network_name):
        # Note: action is the index of the action defined in the environment's action space
        assert(network_name == "target" or network_name == "approx")

        if network_name == "target":
            Q_actions = self.target_network(state)

        elif network_name == "approx":
            Q_actions = self.approx_network(state)

        # action here is an int and not a tensor so we don't need to convert it
        return Q_actions[action] # int(action.numpy())

    # Returns index of action in action space with highest Q value
    def get_best_action(self, state, network_name) -> int:
        if network_name == "target":
            Q_actions = self.target_network(state)
        elif network_name == "approx":
            Q_actions = self.approx_network(state)

        best_action_index = torch.argmax(Q_actions.detach())
        return best_action_index.item()

    #
    def train_on_minibatch(self, minibatch, timestep):
        states, actions, rewards, next_states, dones = minibatch

        # q_vals = self.approx_network.forward(states.unsqueeze(1))        # [batch_size, num_actions]
        # q_chosen = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)    # [batch_size]

        # states_flatten = torch.flatten(states, start_dim=1)
        q_vals = self.approx_network(states)  # states_flatten
        q_chosen = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)

        # TODO: Issue is that the target computations are noisy and unstable because we are still not doing batch updating

        # best_actions = torch.tensor([self.get_best_action(torch.tensor([ns], dtype=torch.double).detach(), "target") for ns in next_states])
        #
        # next_q_values = torch.tensor([
        #     self.target_network(torch.tensor([ns], dtype=torch.double))[a].detach() for ns, a in zip(next_states, best_actions)
        # ], dtype=torch.double)
        #
        # target = torch.where(
        #     dones,
        #     rewards,
        #     rewards + (self.config.gamma * next_q_values)
        # )

        # New version added
        with torch.no_grad():
            # next_states_flatten = torch.flatten(next_states, start_dim=1)
            q_next_all = self.target_network(next_states)  # next_states_flatten [batch_size, num_actions]
            best_actions = torch.argmax(q_next_all, dim=1)  # [batch_size]
            next_q_values = q_next_all.gather(1, best_actions.unsqueeze(1)).squeeze(1)  # [batch_size]

            # Q_target = r if done else r + gamma * max_a Q_target(s', a)
            target = torch.where(
                dones,
                rewards,
                rewards + self.config.gamma * next_q_values
            )

        self.approx_network.optimizer.zero_grad()       # moved reset optimizer to before we compute loss. was always before loss.backward() tho
        self.target_network.optimizer.zero_grad()       # idk if this'll do anything bc we shouldn't need to do anything with target network gradients

        loss = self.approx_network.criterion(q_chosen, target)
        writer.add_scalar("Loss/train", loss.item(), timestep)
        writer.add_scalar("Reward/train", np.average(rewards).item(), timestep)
        loss.backward()

        # add gradient clipping again
        if self.config.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.approx_network.parameters(), self.config.clip_val)


        self.approx_network.optimizer.step()
        self.approx_network.scheduler.step()


    def monitor_performance(self, reward, env, timestep):
        writer.add_scalar("Performance/DummyEnv_State_Before_Terminal_Q(s,a)", self.get_Q_value(torch.tensor([4], dtype=torch.double), 0, "approx"), timestep)


    def set_target_weights(self):
        # target network weights and biases <-- approx network weights and biases
        approx_network_state_dict = self.approx_network.state_dict()
        self.target_network.load_state_dict(approx_network_state_dict)


class LinearNN(nn.Module):

    def linear_decay(self, current_step):
        if current_step >= self.config.lr_n_steps:
            return self.config.lr_end / self.config.lr_begin
        return 1.0 - (current_step / self.config.lr_n_steps) * (1 - self.config.lr_end / self.config.lr_begin)

    def __init__(self, env, config):
        # torch.manual_seed(82)                   # TODO: remove this after implementing
        super(LinearNN, self).__init__()
        self.env = env
        self.config = config
        self.fc1 = nn.Linear(np.prod(self.env.state_shape)*self.config.frame_stack_size, self.env.numActions)

        self.double()       # converts model to double to match datatype of input with no serious performance problems
        self.optimizer = optim.Adam(self.parameters(), lr=self.config.lr_begin)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.linear_decay)
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

    for state_index in range(len(env.states)):

        # state = torch.tensor([state], dtype=torch.double)
        state = torch.tensor(np.array(env.states[state_index]), dtype=torch.double)
        state = model.process_state(state)
        # state = torch.flatten(state)    # using default bc here we're not using a batch     , start_dim=1

        for action in range(len(env.actions)):
            print(state_index, "\t\t", action, "\t\t", " Q(s,a)= ", model.get_Q_value(state, action, "approx"))    # int(state[0].numpy())

    print("==============================================")
    print()
    print("Best trajectory: ")

    # Get best trajectory from state 0
    state = env.reset()
    state = model.process_state(state)

    states_visited = []
    actions_taken = []
    rewards_received = []

    while True:
        # state = torch.flatten(state)                             # using default bc no batch here
        best_action = model.get_best_action(state, "approx")

        states_visited.append(np.average(state))
        actions_taken.append(best_action)

        next_state, reward, done = env.take_action(best_action)
        rewards_received.append(reward)

        state = model.process_state(next_state)

        if done:
            break

    print("Best trajectory from Test Environment")
    for i in range(len(states_visited)):
        s = states_visited[i]
        a = actions_taken[i]
        r = rewards_received[i]

        print("State: ", s, " Action: ", a, " Reward Received: ", r)

    print()
    print("Total Reward Received: ", sum(rewards_received))

    return sum(rewards_received)            # rewards from the best possible trajectory


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
    print(model.target_network(state))

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

def batch_update_test():

    batch_size = 10

    states = torch.tensor([2 for i in range(batch_size)], dtype=torch.double)                # [batch_size]

    actions = torch.tensor([1 for i in range(batch_size)])      # [batch_size]

    env = TestEnv()
    config = LinearConfig()
    model = LinearNN(env, config)
    criterion = nn.MSELoss()

    q_vals = model.forward(states.unsqueeze(1))                  # [batch_size, num_actions]
    q_chosen = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1) # [batch_size]

    target = torch.tensor([3 for i in range(batch_size)])       # [batch_size]

    # print("q_vals: ", q_vals)
    # print("q_chosen: ", q_chosen)

    loss = criterion(q_chosen, target)                         # 0-d scalar

    # print("loss shape: ", loss.shape)
    # print("loss: ", loss)

    print("states shape: ", states.shape)
    print("actions shape: ", actions.shape)
    print("q_vals shape: ", q_vals.shape)
    print("q_chosen shape: ", q_chosen.shape)
    print("target shape: ", target.shape)


# def objective(trial):
def main():

    # batch_update_test()

    # config = LinearConfig(
    #     # nsteps_train = trial.suggest_categorical("nsteps_train", [10000, 11000, 12000, 13000, 14000, 15000]),
    #     lr_begin = trial.suggest_categorical("lr_begin", [0.005, 0.05] ),   # low=0.0005, high=0.001, step=0.0005
    #     # epsilon_decay_percentage = trial.suggest_categorical("epsilon_decay_percentage", [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4]),    # , low=0.3, high=1, step=0.1
    #     # lr_decay_percentage = trial.suggest_categorical("lr_decay_percentage", [0.5, 0.6, 0.7, 0.8, 0.9, 1]),
    #     target_weight_update_freq = trial.suggest_categorical("target_weight_update_freq", [100, 300, 400, 600]),
    #     high = trial.suggest_categorical("high", [150, 250, 350, 450, 550]),
    #     minibatch_size = trial.suggest_categorical("minibatch_size", [32, 64, 128, 256]),
    #     replay_buffer_size = trial.suggest_categorical("replay_buffer_size", [1000, 10000 * 5 * 1000]),
    # )
    # for i in range(20):

    for i in range(10):
        print("Starting Training")
        config = LinearConfig()

        # env = DummyEnv()      # maybe the optimal path is too improbable because the reward of 1 only comes after 9 successive random guesses of taking action index 0 (move right)
        env = TestEnv()         # first see if we can learn the TestEnv with random action

        model = Linear(env, config)
        model.train()
        writer.flush()
        writer.close()

        print("Summary AFTER training")
        summary(model, env, config)
    # writer.add_scalar("Performance/NumTimesSeeOptimalTrajectory", total_rewards, i)



if __name__ == '__main__':
    main()
    #
    # storage_url = "sqlite:///db.sqlite3"
    # study = optuna.create_study(direction="maximize", storage=storage_url, study_name="State rand int rep and normalization", load_if_exists=True)
    # study.optimize(objective, n_trials=150)
    #
    # print("Best Params: ", study.best_params)
    # print("Optimization complete. Data saved to:", storage_url)




