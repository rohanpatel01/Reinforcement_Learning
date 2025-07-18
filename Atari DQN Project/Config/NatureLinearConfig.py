

class NatureLinearConfig:

    def __init__(self, **kwargs):
        self.high                            = 255.

        # TODO: New
        self.gradient_momentum = 0  # momentum
        self.squared_gradient_momentum = 0.9  # 0.95   # alpha (rho)
        self.no_op_max_eval = 30
        self.rms_eps = 0.01

        self.learning_delay                  = 0

        self.nsteps_train                   = 8000

        # TODO: New
        self.eval_freq                      = 9000 # purposefully made higher so we wont
        self.num_episodes_test              = 20
        self.soft_epsilon                   = 0.0

        self.learning_start                  = 200
        self.learning_freq                   = 4

        self.grad_clip                       = True
        self.clip_val                        = 10

        self.lr_begin                        = 0.00025
        self.lr_end                          = 0.0001
        self.lr_decay_percentage             = 0.4
        self.lr_n_steps                      = self.nsteps_train * self.lr_decay_percentage

        self.begin_epsilon                   = 1
        self.end_epsilon                     = 0.01 # 0.01 # 0.01
        self.epsilon_decay_percentage        = 0.5
        self.max_time_steps_update_epsilon   = self.nsteps_train * self.epsilon_decay_percentage

        self.minibatch_size = 32
        self.replay_buffer_size = 500
        self.gamma = 0.99
        self.target_weight_update_freq = 200 # 200 got better performance
        self.frame_stack_size = 1

        for key, value in kwargs.items():
            setattr(self, key, value)

