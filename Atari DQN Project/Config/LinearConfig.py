

class LinearConfig:

    def __init__(self, **kwargs):

        self.learning_delay                  = 0         # number of timesteps to perform random actions for

        self.nsteps_train                    = 10000
        self.num_episodes_test               = 20

        self.learning_start                  = 200
        self.learning_freq                   = 4

        self.grad_clip                       = False
        self.clip_val                        = 10


        self.lr_begin                        = 0.05 # best performance was 0.05 I think bc full DummyEnv got 45% accuract and Smaller for 80%
        self.lr_end                          = 0.001 # 0.001
        self.lr_decay_percentage             = 0.5
        self.lr_n_steps                      = self.nsteps_train * self.lr_decay_percentage# * 0.5 # was be // 2 ???
        # step_size                       = int(nsteps_train/3)       # every 10,000 time steps we will update the learning rate   #????? correct??

        self.begin_epsilon                   = 1
        self.end_epsilon                     = 0.01
        self.epsilon_decay_percentage        = 0.7
        self.max_time_steps_update_epsilon   = self.nsteps_train * self.epsilon_decay_percentage

        self.minibatch_size = 128       # TODO: Theory is that if early on the buffer contains small number of samples we will draw minibatch of same / similar experiences thus leading to iid breakdown and instability in training
        self.replay_buffer_size = self.nsteps_train * 5 * 1000  # they use 1000   - try changing between large buffer and 1000 size buffer (basically only last 200 episodes of experience)
        self.gamma = 1
        self.target_weight_update_freq = 500  # was 500         # maybe try what they have: 200 later bc 200 seems too small
        self.frame_stack_size = 1  # For TestEnv this will be 1

        for key, value in kwargs.items():
            setattr(self, key, value)

