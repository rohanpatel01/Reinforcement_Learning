

class AtariDQNConfig:


    def __init__(self, **kwargs):

        self.high = 255.
        self.learning_delay = 0  # TODO: remove this everywhere once done

        self.num_episodes_test = 5  # we will record this many videos during evaluation# 50   Note: just for development we will change some of these values
        self.grad_clip = True
        self.clip_val = 10
        self.saving_freq = 250000
        self.log_freq = 50
        self.eval_freq = 250000
        self.record_freq = 250000   # Note: the way I coded it, for a recording to be made record_freq must equal eval_freq or at least be a common multiple. otherwise recording will not be made
        self.soft_epsilon = 0.05

        # nature paper hyper params
        self.nsteps_train = 5000000 # 5M
        self.minibatch_size = 32  # batch_size
        self.replay_buffer_size = 1000000  # orig is just buffer_size
        self.target_weight_update_freq = 10000  # target_update_freq
        self.gamma = 0.99
        self.learning_freq = 4
        self.state_history = 4
        self.skip_frame = 4
        self.lr_begin = 0.00025
        self.lr_end = 0.00005
        self.lr_n_steps = self.nsteps_train / 2  # orig was lr_nsteps
        self.begin_epsilon = 1  # eps_begin
        self.end_epsilon = 0.1  # eps_end
        self.max_time_steps_update_epsilon = 1000000  # eps_nsteps
        self.learning_start = 50000



        for key, value in kwargs.items():
            setattr(self, key, value)