import copy

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.cuda import device
from torch.utils.tensorboard import SummaryWriter
import time

from typing_extensions import override

from Q_Learning import Q_Learning
from TestEnv import TestEnv
from DummyEnv import DummyEnv
from SimpleEnv import SimpleEnv
from Config.LinearConfig import LinearConfig
from ReplayBuffer import ReplayBuffer
from Scheduler import EpsilonScheduler
import optuna
from optuna_dashboard import run_server
import time

import gymnasium as gym

from gymnasium.wrappers import (
    GrayscaleObservation,
    ResizeObservation,
    FrameStackObservation, RecordVideo,
    AtariPreprocessing
)
import os

writer = SummaryWriter()
video_dir = "videos/"
os.makedirs(video_dir, exist_ok=True)

from gymnasium.spaces import Discrete


class ReducedActionSet(gym.ActionWrapper):
    def __init__(self, env, allowed_actions):
        super().__init__(env)
        self.allowed_actions = allowed_actions
        self.action_space = Discrete(len(self.allowed_actions))

    def action(self, action):
        return self.allowed_actions[action]


class Linear(Q_Learning):

    def __init__(self, env, config, device):
        super(Linear, self).__init__(env, config, device)
        self.env = env
        self.config = config
        self.device = device
        self.target_network = LinearNN(env, config).to(device)
        self.approx_network = LinearNN(env, config).to(device)
        self.set_target_weights()                   # Initially we want both target and approx network to have the same arbitrary weights

        self.t = 1
        self.t = 1  # have our own time so we can load it from disk
        self.replay_buffer = ReplayBuffer(self.config.replay_buffer_size, self.env, self.config)  # have our own replay buffer so that we can load it in from disk
        self.num_episodes = 0
        self.total_reward_so_far = 0



    def sample_action(self, env, state, epsilon, time, network_name):

        if  (time < self.config.learning_delay )or  (np.random.rand() < epsilon):
            # take random action
            return np.random.randint(env.action_space.n)  # numActions        action_space.n
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
        # state = state.to(self.device)
        if network_name == "target":
            Q_actions = self.target_network(state)
        elif network_name == "approx":
            Q_actions = self.approx_network(state)

        best_action_index = torch.argmax(Q_actions.detach())
        return best_action_index.item()

    #
    def train_on_minibatch(self, minibatch, timestep):
        states, actions, rewards, next_states, dones = minibatch

        states = self.process_state(states)
        next_states = self.process_state(next_states)

        states = states.to(self.device).to(torch.float32)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device).to(torch.float32)
        dones = dones.to(self.device)

        q_vals = self.approx_network(states)
        q_chosen = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)

        # New version added
        with torch.no_grad():
            q_next_all = self.target_network(next_states)  # [batch_size, num_actions]
            best_actions = torch.argmax(q_next_all, dim=1)  # [batch_size]
            next_q_values = q_next_all.gather(1, best_actions.unsqueeze(1)).squeeze(1)  # [batch_size]

            # Q_target = r if done else r + gamma * max_a Q_target(s', a)
            target = torch.where(
                dones,
                rewards,
                rewards + self.config.gamma * next_q_values # TODO: new - removed parenthesis
            )

        self.approx_network.optimizer.zero_grad()       # moved reset optimizer to before we compute loss. was always before loss.backward() tho
        self.target_network.optimizer.zero_grad()       # idk if this'll do anything bc we shouldn't need to do anything with target network gradients

        loss = self.approx_network.criterion(q_chosen, target)
        writer.add_scalar("Loss/train", loss.item(), timestep)
        writer.add_scalar("Reward/train", torch.mean(rewards), timestep)
        loss.backward()

        # Monitor the model and check for vanishing gradients due to weights -> 0
        writer.add_scalar("Model/Conv1_Weight_Gradients", self.approx_network.conv1.weight.grad.abs().mean(), timestep)
        writer.add_scalar("Model/Conv2_Weight_Gradients", self.approx_network.conv2.weight.grad.abs().mean(), timestep)
        writer.add_scalar("Model/Conv3_Weight_Gradients", self.approx_network.conv3.weight.grad.abs().mean(), timestep)
        writer.add_scalar("Model/fc1_Weight_Gradients", self.approx_network.fc1.weight.grad.abs().mean(), timestep)
        writer.add_scalar("Model/fc2_Weight_Gradients", self.approx_network.fc2.weight.grad.abs().mean(), timestep)

        writer.add_scalar("Model/Conv1_bias_Gradients", self.approx_network.conv1.bias.grad.abs().mean(), timestep)
        writer.add_scalar("Model/Conv2_bias_Gradients", self.approx_network.conv2.bias.grad.abs().mean(), timestep)
        writer.add_scalar("Model/Conv3_bias_Gradients", self.approx_network.conv3.bias.grad.abs().mean(), timestep)
        writer.add_scalar("Model/fc1_bias_Gradients", self.approx_network.fc1.bias.grad.abs().mean(), timestep)
        writer.add_scalar("Model/fc2_bias_Gradients", self.approx_network.fc2.bias.grad.abs().mean(), timestep)

        # Target network weights
        writer.add_scalar("Model/Conv1_Weight_Gradients", self.target_network.conv1.weight.abs().mean(), timestep)
        writer.add_scalar("Model/Conv2_Weight_Gradients", self.target_network.conv2.weight.abs().mean(), timestep)
        writer.add_scalar("Model/Conv3_Weight_Gradients", self.target_network.conv3.weight.abs().mean(), timestep)
        writer.add_scalar("Model/fc1_Weight_Gradients", self.target_network.fc1.weight.abs().mean(), timestep)
        writer.add_scalar("Model/fc2_Weight_Gradients", self.target_network.fc2.weight.abs().mean(), timestep)

        writer.add_scalar("Model/Conv1_bias_Gradients", self.target_network.conv1.bias.abs().mean(), timestep)
        writer.add_scalar("Model/Conv2_bias_Gradients", self.target_network.conv2.bias.abs().mean(), timestep)
        writer.add_scalar("Model/Conv3_bias_Gradients", self.target_network.conv3.bias.abs().mean(), timestep)
        writer.add_scalar("Model/fc1_bias_Gradients", self.target_network.fc1.bias.abs().mean(), timestep)
        writer.add_scalar("Model/fc2_bias_Gradients", self.target_network.fc2.bias.abs().mean(), timestep)

        # add gradient clipping again
        if self.config.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.approx_network.parameters(), self.config.clip_val)

        self.approx_network.optimizer.step()
        self.approx_network.scheduler.step()


    def monitor_performance(self, state, reward, monitor_end_of_episode, timestep, context=None):

        with torch.no_grad():

            if monitor_end_of_episode:
                assert(context is not None)

                total_reward_so_far, num_episodes, total_reward_for_episode = context
                writer.add_scalar("Evaluation/Avg_Reward", total_reward_so_far / num_episodes, self.t)
                # Assuming (might be wrong) that by Max_Reward they mean max reward seen per episode, which would just be the reward at end of episode
                writer.add_scalar("Evaluation/Max_Reward", total_reward_for_episode, self.t)
                return

            # Note: episode has not terminated
            state = state.to(self.device)
            q_action_vals = self.approx_network(state)

            writer.add_scalar("Evaluation/STDV", torch.std(q_action_vals), timestep)    # measure the standard deviation - it shouldn't be too small bc otherwise means all states have similar q values (not good)
            writer.add_scalar("Evaluation/Max_Q", torch.max(q_action_vals), timestep)


            # Note: we only evaluate/record if it's not the end of an episode (above check handles assuring this)
            # This is necessary bc we want to check every timestep if we need to evaluate, otherwise we may miss it bc current episode hasn't terminated
            if timestep % self.config.eval_freq == 0:
                # TODO: Use state to track the max_q when we do a forward pass
                record_env = gym.make("ALE/Pong-v5", obs_type="rgb", render_mode="rgb_array", frameskip=4, repeat_action_probability=0)   # rgb_array needed for video recording
                record_env = ReducedActionSet(record_env, allowed_actions=[0, 2, 3])
                record_env = FrameStackObservation(record_env, stack_size=4)
                record_env = RecordVideo(
                    record_env,
                    video_folder="pong",
                    name_prefix="eval",
                    episode_trigger=lambda x: (x==0) and ((timestep % self.config.record_freq) == 0)      # internally, the env has a counter and it consults episode_trigger's boolean value of whether it should record that episode or not
                )


                eval_env = gym.make("ALE/Pong-v5", frameskip=1, repeat_action_probability=0)
                eval_env = ReducedActionSet(eval_env, allowed_actions=[0, 2, 3])
                eval_env = AtariPreprocessing(
                    eval_env,
                    noop_max=self.config.no_op_max_eval, frame_skip=4, terminal_on_life_loss=False,         # use noop max of 30 during evaluation
                    screen_size=84, grayscale_obs=True, grayscale_newaxis=False,    # If error pops up, make sure screen size for eval_env matches env (from DQN) or (from Linear)
                    scale_obs=False
                )
                eval_env = FrameStackObservation(eval_env, stack_size=4)
                total_reward = 0

                for episode in range(self.config.num_episodes_test):

                    # Note: Using same seed so that states and actions of Record env and eval_env match so that
                    # we see exactly the states the agent performs the actions in - but just that we record in color and not the preprocessed version we pass to the NN
                    rand_seed = np.random.randint(100)
                    record_env.reset(seed=rand_seed)

                    state, info = eval_env.reset(seed=rand_seed)  # seed=rand_seed
                    state = torch.from_numpy(state)
                    state = self.process_state(state)
                    state = state.to(self.device)

                    while True:
                        with torch.no_grad():

                            action = self.sample_action(eval_env, state,
                                                        self.config.soft_epsilon,
                                                        self.t, "approx")

                            # next_state, reward, done = self.env.take_action(action)
                            next_state, reward, terminated, truncated, info = eval_env.step(action)
                            record_env.step(action)

                            next_state = torch.from_numpy(next_state)
                            next_state = self.process_state(next_state).to(self.device)
                            state = next_state

                            total_reward += reward
                            if terminated:
                                break

                # Done with episodes
                # We are just monitoring the number of win/loss (+1, -1) to monitor the Eval_reward
                writer.add_scalar("Evaluation/Avg_Eval_Reward", total_reward/self.config.num_episodes_test, timestep)
                record_env.close()  # close to flush video to file

    def save_snapshop(self, timestep, num_episodes, total_reward_so_far, replay_buffer):
        snapshot = {
            "timestep" : self.t,
            "num_episodes" : num_episodes,
            "total_reward_so_far" : total_reward_so_far,
            "replay_buffer_list" : replay_buffer.replay_buffer,
            "replay_buffer_next_replay_location" : replay_buffer.next_replay_location,
            "replay_buffer_num_elements" : replay_buffer.num_elements,
            "approx_network_state_dict" : self.approx_network.state_dict(),
            "target_network_state_dict" : self.target_network.state_dict()
        }

        torch.save(snapshot, "snapshot.pt")


    def load_snapshot(self):

        if not os.path.exists("snapshot.pt"):
            print("Attempted to load snapshot which does not exist - skipping")
            return

        snapshot = torch.load("snapshot.pt", map_location='cpu', weights_only=False)

        self.t = snapshot["timestep"]
        self.num_episodes = snapshot["num_episodes"]
        self.total_reward_so_far = snapshot["total_reward_so_far"]

        self.replay_buffer.replay_buffer = snapshot["replay_buffer_list"]
        self.replay_buffer.next_replay_location = snapshot["replay_buffer_next_replay_location"]
        self.replay_buffer.num_elements = snapshot["replay_buffer_num_elements"]

        self.approx_network.load_state_dict(snapshot["approx_network_state_dict"])
        self.target_network.load_state_dict(snapshot["target_network_state_dict"])


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
        self.fc1 = nn.Linear(np.prod(self.env.observation_space.shape), self.env.action_space.n )            # observation_space.shape    action_space.n               self.env.state_shape   self.env.numActions

        self.float()       # converts model to double to match datatype of input with no serious performance problems
        self.optimizer = optim.Adam(self.parameters(), lr=self.config.lr_begin)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.linear_decay)
        self.criterion = nn.MSELoss()
        # print("layer weights: ", self.fc1.weight)


    def forward(self, x):

        if len(x.shape) == 3:
            x = torch.flatten(x)
        elif len(x.shape) == 4:
            x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        return x


def summary(model, env, config):

    state = env.reset()
    device = model.device

    print()
    print("==============================================")
    print("Summary: ")
    # Print out all Q(s,a) values
    print("State\tAction\tNext State\tReward")

    for state_index in range(len(env.states)):

        state = torch.tensor(np.array(env.states[state_index]), dtype=torch.float32)
        state = model.process_state(state)
        state = state.to(device)
        for action in range(len(env.actions)):
            print(state_index, "\t\t", action, "\t\t", " Q(s,a)= ", model.get_Q_value(state, action, "approx"))    # int(state[0].numpy())

    print("==============================================")
    print()
    print("Best trajectory: ")

    # Get best trajectory from state 0
    state, _ = env.reset()
    state = model.process_state(state)
    state = state.to(device)

    states_visited = []
    actions_taken = []
    rewards_received = []

    while True:
        best_action = model.get_best_action(state, "approx")

        states_visited.append(torch.mean(state))
        actions_taken.append(best_action)

        next_state, reward, done, _, _ = env.step(best_action)
        next_state = model.process_state(next_state).to(device)

        rewards_received.append(reward)

        # state = model.process_state(next_state)
        state = next_state


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
    # How to connect to GPU
    print("Pytorch version: ", torch.__version__)
    print("Number of GPU: ", torch.cuda.device_count())
    print("GPU Name: ", torch.cuda.get_device_name())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # Note: for smaller environments (e.g (5,5,1)) cpu is faster (~5s cpu vs ~15s gpu) but larger envs (e.g (80,80,1)) the gpu is faster. This is likely due to having to move data from cpu to gpu takes longer and by comparison the compute time for small envs is neglegible
    print('Using device:', device)

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

    for i in range(1):
        print("Starting Training")
        config = LinearConfig()


        env = TestEnv((5,5,1))         # first see if we can learn the TestEnv with random action

        model = Linear(env, config, device)

        start_time = time.time()
        model.train()
        end_time = time.time()

        elapsed_time = end_time - start_time

        print(f"Elapsed time: {elapsed_time:.6f} seconds")

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





