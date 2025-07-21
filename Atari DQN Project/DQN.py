
import torch
import torch.nn as nn
from torch import optim
from Config.NatureLinearConfig import NatureLinearConfig
from Config.AtariLinearConfig import AtariLinearConfig
from Config.AtariDQNConfig import AtariDQNConfig
from TestEnv import TestEnv
import Linear
from Linear import Linear
from Linear import writer
from Linear import summary
import optuna
from optuna_dashboard import run_server
from DummyEnv import DummyEnv
import time
import ale_py
import gymnasium as gym
from ReplayBuffer import ReplayBuffer
from datetime import datetime
import os

from Linear import ReducedActionSet

# Note: You can access the environment underneath the first wrapper by using the gymnasium.Wrapper.env attribute.
#       If you want to get to the environment underneath all of the layers of wrappers, you can use the gymnasium.Wrapper.unwrapped attribute

from gymnasium.wrappers import (
    AtariPreprocessing,
    GrayscaleObservation,
    ResizeObservation,
    FrameStackObservation,
    RecordVideo
)



class DQN(Linear):

    def __init__(self, env, config, device):
        super(DQN, self).__init__(env, config, device)
        self.env = env
        self.config = config
        self.device = device

        self.target_network = NatureQN(env, config, device).to(device)
        self.approx_network = NatureQN(env, config, device).to(device)

        self.set_target_weights()

        self.t = 1                  # have our own time so we can load it from disk
        self.replay_buffer = ReplayBuffer(self.config.replay_buffer_size, self.env, self.config)       # have our own replay buffer so that we can load it in from disk
        self.num_episodes = 0
        self.total_reward_so_far = 0


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

        snapshot = torch.load("snapshot.pt")

        self.t = snapshot["timestep"]
        self.num_episodes = snapshot["num_episodes"]
        self.total_reward_so_far = snapshot["total_reward_so_far"]

        self.replay_buffer.replay_buffer = snapshot["replay_buffer_list"]
        self.replay_buffer.next_replay_location = snapshot["replay_buffer_next_replay_location"]
        self.replay_buffer.num_elements = snapshot["replay_buffer_num_elements"]

        self.approx_network.load_state_dict(snapshot["approx_network_state_dict"])
        self.target_network.load_state_dict(snapshot["target_network_state_dict"])

class NatureQN(nn.Module):

    def linear_decay(self, current_step):
        if current_step >= self.config.lr_n_steps:
            return self.config.lr_end / self.config.lr_begin
        return 1.0 - (current_step / self.config.lr_n_steps) * (1 - self.config.lr_end / self.config.lr_begin)


    # Following Model Architecture from mnih2015human
    def __init__(self, env, config, device):
        # torch.manual_seed(42)             # for testing purposes
        super(NatureQN, self).__init__()
        self.env = env
        self.config = config
        self.device = device

        # Note default data type for nn.Conv2d weights and biases are float so we change to float32 to match state datatype
        self.conv1 = nn.Conv2d(in_channels=env.observation_space.shape[0], out_channels=32, kernel_size=8, stride=4, dtype=torch.float32)  #   state_shape[0]     observation_space.shape[0]
        self.bn_conv1 = nn.BatchNorm2d(32)  # we're normalizing the input to the next layer so match next layer's num in_channels

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, dtype=torch.float32)
        self.bn_conv2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, dtype=torch.float32)
        self.bn_conv3 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(in_features=3136, out_features=512, dtype=torch.float32)   # 2304 for (4, 80, 80)          3136 for (4, 84, 84)       4 for Cartpole
        self.bn_fc1 = nn.BatchNorm1d(512)

        self.fc2    = nn.Linear(in_features=512, out_features=env.action_space.n, dtype=torch.float32)  # 512  25             # action_space.n      numActions


        self.ReLU = nn.ReLU()

        self.optimizer = optim.RMSprop(self.parameters(), lr=self.config.lr_begin, alpha=self.config.squared_gradient_momentum, eps=self.config.rms_eps)  # , alpha=self.config.squared_gradient_momentum, eps=self.config.rms_eps
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.linear_decay)
        # self.criterion = nn.MSELoss()

        self.criterion = nn.SmoothL1Loss()


    def forward(self, x):

        # if len(x.shape) == 4:
        #     x = self.bn_conv1(self.conv1(x))
        # elif len(x.shape) == 3:
        #     x = self.conv1(x)
        #
        # x = self.ReLU(x)
        #
        # if len(x.shape) == 4:
        #     x = self.bn_conv2(self.conv2(x))
        # elif len(x.shape) == 3:
        #     x = self.conv2(x)
        #
        # x = self.ReLU(x)
        #
        #
        # if len(x.shape) == 4:
        #     x = self.bn_conv3(self.conv3(x))
        # elif len(x.shape) == 3:
        #     x = self.conv3(x)
        #
        # x = self.ReLU(x)
        #
        # # un-permute before we flatten
        # if len(x.shape) == 3:   # single input [height, width, num_channels]
        #     x = torch.flatten(x)
        # elif len(x.shape) == 4:  # batch input [batch_size, height, width, num_channels]
        #     x = torch.flatten(x, start_dim=1)
        #
        # if len(x.shape) == 2:
        #     x = self.bn_fc1(self.fc1(x))
        # elif len(x.shape) == 1:
        #     x = self.fc1(x)
        #
        #
        # x = self.ReLU(x)
        # x = self.fc2(x)

        x = self.conv1(x)
        x = self.ReLU(x)

        x = self.conv2(x)
        x = self.ReLU(x)

        x = self.conv3(x)
        x = self.ReLU(x)

        if len(x.shape) == 3:  # single input [height, width, num_channels]
            x = torch.flatten(x)
        elif len(x.shape) == 4:  # batch input [batch_size, height, width, num_channels]
            x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = self.ReLU(x)
        x = self.fc2(x)

        return x

# def objective(trial):
def main():


    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print('Using device:', device)


    # MAX_REWARD = 4.1
    # count_max_reward = 0
    num_trials_test = 1


    for i in range(num_trials_test):
        print("starting trial: ", i+1)

        # config = NatureLinearConfig()
        # config = AtariLinearConfig()
        config = AtariDQNConfig()
        # print(gym.envs.registration.registry.keys())
        env = gym.make("ALE/Pong-v5", frameskip=1, repeat_action_probability=0)
        env = ReducedActionSet(env, allowed_actions=[0, 2, 3])
        env = AtariPreprocessing(
            env,
            noop_max=30, frame_skip=4, terminal_on_life_loss=False, # changed noop_max to 30 from 0
            screen_size=84, grayscale_obs=True, grayscale_newaxis=False,
            scale_obs=False
        )
        env = FrameStackObservation(env, stack_size=4)

        model = DQN(env, config, device)
        # model = Linear(env, config, device) # first test atari env with Linear model since train time for that is just 1 hour and we can confirm that it doesn't learn well. So hopefully things are "working" there
        # model.load_snapshot() # just right now I don't want to load the snapshot
        # start_time = time.time()
        model.train()
        # end_time = time.time()

        writer.flush()
        writer.close()

        # total_reward = summary(model, env, config)
        # if total_reward == MAX_REWARD:
        #     count_max_reward += 1


        # elapsed_time = end_time - start_time

        # print(f"Elapsed time: {elapsed_time:.6f} seconds")

    # print("Training stability: ", count_max_reward / num_trials_test)






if __name__ == '__main__':

    main()

    # storage_url = "sqlite:///db.sqlite3"
    # study = optuna.create_study(direction="maximize", storage=storage_url, study_name="Find_Best_Params_DQN_TestEnv_%_Success_Test_overnight")
    # study.optimize(objective, n_trials=200)
    #
    # print("Best Params: ", study.best_params)
    # print("Optimization complete. Data saved to:", storage_url)