

class NatureLinearConfig:

    def __init__(self, **kwargs):
        self.high                            = 255.

        self.learning_delay                  = 0

        self.nsteps_train                    = 1000

        self.learning_start                  = 200
        self.learning_freq                   = 4

        self.grad_clip                       = True
        self.clip_val                        = 10

        self.lr_begin                        = 0.00025
        self.lr_end                          = 0.0001
        self.lr_decay_percentage             = 0.5
        self.lr_n_steps                      = self.nsteps_train * self.lr_decay_percentage

        self.begin_epsilon                   = 1
        self.end_epsilon                     = 0.01 # 0.01
        self.epsilon_decay_percentage        = 0.5
        self.max_time_steps_update_epsilon   = self.nsteps_train * self.epsilon_decay_percentage

        self.minibatch_size = 32
        self.replay_buffer_size = 500
        self.gamma = 0.99
        self.target_weight_update_freq = 500
        self.frame_stack_size = 1

        for key, value in kwargs.items():
            setattr(self, key, value)

