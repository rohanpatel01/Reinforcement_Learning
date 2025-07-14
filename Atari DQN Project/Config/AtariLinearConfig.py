
class AtariLinearConfig:


    def __init__(self, **kwargs):

        high = 255.

        grad_clip = True
        clip_val = 10

        nsteps_train = 5000000
        batch_size = 32
        buffer_size = 1000000
        target_update_freq = 10000
        gamma = 0.99
        learning_freq = 4
        state_history = 4
        skip_frame = 4
        lr_begin = 0.00025
        lr_end = 0.00005
        lr_nsteps = nsteps_train / 2
        eps_begin = 1
        eps_end = 0.1
        eps_nsteps = 1000000
        learning_start = 50000

        for key, value in kwargs.items():
            setattr(self, key, value)