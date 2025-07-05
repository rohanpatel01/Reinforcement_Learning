

class LinearConfig:

    def __init__(self, **kwargs):

        self.learning_start                  = 200
        self.learning_freq                   = 4

        self.grad_clip                       = True
        self.clip_val                        = 10

        self.nsteps_train                    = 3000

        self.lr_begin                        = 0.0005 #0.005    bigest change was making learning rate smaller - before was 0.005 which showed loss function as unstable and increasing
        self.lr_end                          = self.lr_begin  # 0.001  # for optuna experiment we will make them the same
        self.lr_n_steps                      = self.nsteps_train / 2
        # step_size                       = int(nsteps_train/3)       # every 10,000 time steps we will update the learning rate   #????? correct??

        self.begin_epsilon                   = 1
        self.end_epsilon                     = 0.01
        self.epsilon_decay_percentage        = 0.6
        self.max_time_steps_update_epsilon   = self.nsteps_train * self.epsilon_decay_percentage

        self.minibatch_size                  = 128
        self.replay_buffer_size              = self.nsteps_train
        self.gamma                           = 0.999
        self.target_weight_update_freq       = 200 # was 500               # change to 4 or 5 later and see if it makes a difference - why do we not update the weights with every experience? Why do we not always have this as 1?
        self.frame_stack_size                = 1                        # For TestEnv this will be 1

        for key, value in kwargs.items():
            setattr(self, key, value)

