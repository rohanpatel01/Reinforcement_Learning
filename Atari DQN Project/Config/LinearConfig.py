

class LinearConfig:

    learning_start                  = 200
    learning_freq                   = 4

    nsteps_train                    = 2500
    num_episodes                    = int(nsteps_train/3)     # 5 bc each episode lasts 5 time steps in TestEnv
    lr_begin                        = 1e-4 #0.005    biggest change was making learning rate smaller - before was 0.005 which showed loss function as unstable and increasing
    lr_end                          = 0.001
    step_size                       = int(nsteps_train/3)       # every 10,000 time steps we will update the learning rate   #????? correct??

    begin_epsilon                   = 1
    end_epsilon                     = 0.01
    max_time_steps_update_epsilon   = int(nsteps_train/2)

    minibatch_size                  = 32
    replay_buffer_size = int(1e6)
    gamma                           = 0
    target_weight_update_freq       = 500 # was 500               # change to 4 or 5 later and see if it makes a difference - why do we not update the weights with every experience? Why do we not always have this as 1?
    frame_stack_size                = 1                        # For TestEnv this will be 1


