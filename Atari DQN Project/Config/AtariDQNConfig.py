

class AtariDQNConfig:


    def __init__(self, **kwargs):

        self.high = 255.
        self.learning_delay = 0
        self.gradient_momentum = 0              # momentum
        self.squared_gradient_momentum = 0.9 # 0.95   # alpha (rho)
        self.no_op_max_eval = 30
        self.rms_eps = 0.01

        self.num_episodes_test = 5  # we will record this many videos during evaluation# 50   Note: just for development we will change some of these values
        self.grad_clip = True
        self.clip_val = 10
        self.saving_freq = 250000
        self.log_freq = 50
        self.eval_freq = 5000 # 50000
        self.record_freq = 5000
        self.soft_epsilon = 0.05

        # nature paper hyper params
        self.nsteps_train = 5000000 # 5M
        self.minibatch_size = 32  # batch_size
        self.replay_buffer_size = 1000000
        self.target_weight_update_freq = 10000
        self.gamma = 0.99
        self.learning_freq = 4
        self.state_history = 4
        self.skip_frame = 4
        self.lr_begin = 0.00025 # orig 0.00025
        self.lr_end = 0.00005   # orig 0.00005
        self.lr_n_steps = self.nsteps_train / 2  # orig was lr_nsteps
        self.begin_epsilon = 1
        self.end_epsilon = 0.1
        self.max_time_steps_update_epsilon = 1000000
        self.learning_start = 30000 # orig 50000



        for key, value in kwargs.items():
            setattr(self, key, value)