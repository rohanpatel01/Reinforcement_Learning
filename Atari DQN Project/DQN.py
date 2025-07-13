
import torch
import torch.nn as nn
from torch import optim
from Config.NatureLinearConfig import NatureLinearConfig
from TestEnv import TestEnv
import Linear
from Linear import Linear
from Linear import writer
from Linear import summary
import optuna
from optuna_dashboard import run_server
from DummyEnv import DummyEnv

# How to connect to GPU
# print("Pytorch version: ", torch.__version__)
# print("Number of GPU: ", torch.cuda.device_count())
# print("GPU Name: ", torch.cuda.get_device_name())
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using device:', device)

class DQN(Linear):

    def __init__(self, env, config):
        super(DQN, self).__init__(env, config)
        self.env = env
        self.config = config
        self.target_network = NatureQN(env, config)
        self.approx_network = NatureQN(env, config)
        self.set_target_weights()



class NatureQN(nn.Module):

    def linear_decay(self, current_step):
        if current_step >= self.config.lr_n_steps:
            return self.config.lr_end / self.config.lr_begin
        return 1.0 - (current_step / self.config.lr_n_steps) * (1 - self.config.lr_end / self.config.lr_begin)


    # Following Model Architecture from mnih2015human
    def __init__(self, env, config):
        # torch.manual_seed(42)             # for testing purposes
        super(NatureQN, self).__init__()
        self.env = env
        self.config = config

        # self.double()

        # Note default data type for nn.Conv2d weights and biases are float so we change to double to match state datatype
        self.conv1 = nn.Conv2d(in_channels=env.state_shape[-1], out_channels=32, kernel_size=8, stride=4, dtype=torch.double) # for Atari the in_channels will be 4 but for TestEnv it will be 1 bc we're not stacking
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, dtype=torch.double)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, dtype=torch.double)
        self.fc1 = nn.Linear(in_features=2304, out_features=512, dtype=torch.double)
        self.fc2    = nn.Linear(in_features=512, out_features=env.numActions, dtype=torch.double)  # 512  25

        self.ReLU = nn.ReLU()

        self.optimizer = optim.RMSprop(self.parameters(), lr=self.config.lr_begin)  # mnih2015human uses RMSprop
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.linear_decay)
        self.criterion = nn.MSELoss()       # we're doing regression to get Q values so MSE is ok to use    - also by using MSE we assume data was sampled from gaussian distribution

    def forward(self, x):


        # permute from NHWC into NCHW format for nn.Conv2d based on batch or single input
        if len(x.shape) == 3:   # single input [height, width, num_channels]
            x = x.permute(2, 0, 1)
        elif len(x.shape) == 4:  # batch input [batch_size, height, width, num_channels]
            x = x.permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.ReLU(x)
        x = self.conv2(x)
        x = self.ReLU(x)
        x = self.conv3(x)
        x = self.ReLU(x)

        # un-permute before we flatten - maybe that is causing values to be weird
        if len(x.shape) == 3:   # single input [height, width, num_channels]
            x = torch.flatten(x)
        elif len(x.shape) == 4:  # batch input [batch_size, height, width, num_channels]
            x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = self.ReLU(x)
        x = self.fc2(x)
        return x

# def objective(trial):
def main():

    # config = NatureLinearConfig(
    #     nsteps_train = trial.suggest_categorical("nsteps_train", [3000, 4000, 5000, 6000, 7000, 8000]),
    #     target_weight_update_freq = trial.suggest_categorical("target_weight_update_freq", [100, 150, 200, 500]),
    #     replay_buffer_size = trial.suggest_categorical("replay_buffer_size", [500, 1000]),
    #     epsilon_decay_percentage = trial.suggest_categorical("epsilon_decay_percentage", [0.4, 0.5, 0.7, 0.9]),
    #     lr_begin = trial.suggest_categorical("lr_begin", [0.00025]),
    #     lr_decay_percentage = trial.suggest_categorical("lr_decay_percentage", [0.4, 0.5, 0.7, 0.9]),
    # )


    MAX_REWARD = 4.1
    count_max_reward = 0
    num_trials_test = 1


    for i in range(num_trials_test):
        config = NatureLinearConfig()
        env = TestEnv((80, 80, 1))

        model = DQN(env, config)
        model.train()
        writer.flush()
        writer.close()

        total_reward = summary(model, env, config)
        if total_reward == MAX_REWARD:
            count_max_reward += 1

    print("Training stability: ", count_max_reward / num_trials_test)






if __name__ == '__main__':
    main()
    # storage_url = "sqlite:///db.sqlite3"
    # study = optuna.create_study(direction="maximize", storage=storage_url, study_name="Find_Best_Params_DQN_TestEnv_%_Success_Test_overnight")
    # study.optimize(objective, n_trials=200)
    #
    # print("Best Params: ", study.best_params)
    # print("Optimization complete. Data saved to:", storage_url)