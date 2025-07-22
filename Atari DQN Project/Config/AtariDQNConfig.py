

class AtariDQNConfig:


    def __init__(self, **kwargs):

        self.high = 255.
        self.learning_delay = 100_000
        self.gradient_momentum = 0              # momentum
        self.squared_gradient_momentum = 0.9    # alpha (rho)
        self.no_op_max_eval = 30
        self.rms_eps = 0.01

        self.num_episodes_test = 5
        self.grad_clip = True
        self.clip_val = 10
        self.saving_freq = 250_000
        self.log_freq = 50
        self.eval_freq = 50000
        self.record_freq = 50000
        self.soft_epsilon = 0.05

        # nature paper hyper params
        self.nsteps_train = 50_000_000      # training for much longer (50 million timesteps)
        self.minibatch_size = 32
        self.replay_buffer_size = 1_000_000
        self.target_weight_update_freq = 10_000
        self.gamma = 0.99
        self.learning_freq = 4
        self.state_history = 4
        self.skip_frame = 4
        self.lr_begin = 0.00025
        self.lr_end = 0.00005
        self.lr_n_steps = self.nsteps_train / 2
        self.begin_epsilon = 1
        self.end_epsilon = 0.1
        self.max_time_steps_update_epsilon = 5_000_000
        self.learning_start = 50000

        for key, value in kwargs.items():
            setattr(self, key, value)