

class LinearConfig:

    learning_start                  = 200
    learning_freq                   = 4

    # num_episodes                    = [1000, 1500, 2000, 2500, 3000, 4000]   # , 1000, 1500, 2000  # 5 bc each episode lasts 5 time steps in TestEnv
    # nsteps_train                    = [5 * 0.1, 5 * 0.2, 5 * 0.3, 5 * 0.4, 5 * 0.5, 5 * 0.6,  5 * 0.7,  5 * 0.8, 5 * 0.9, 5 * 1, 2, 3, 4]          # higher the decimal means more exploration        smaller decimal means more exploitation
    num_episodes                    = 2500
    nsteps_train                    = num_episodes * 2

    lr_begin                        = 1e-3 #0.005    biggest change was making learning rate smaller - before was 0.005 which showed loss function as unstable and increasing
    lr_end                          = 0.001
    # step_size                       = int(nsteps_train/3)       # every 10,000 time steps we will update the learning rate   #????? correct??

    begin_epsilon                   = 1
    end_epsilon                     = 0.01 #0.01
    max_time_steps_update_epsilon   = float('-inf') # int(nsteps_train)

    minibatch_size                  = 64
    replay_buffer_size = int(1e4)
    gamma                           = 1
    target_weight_update_freq       = 200 # was 500               # change to 4 or 5 later and see if it makes a difference - why do we not update the weights with every experience? Why do we not always have this as 1?
    frame_stack_size                = 1                        # For TestEnv this will be 1


