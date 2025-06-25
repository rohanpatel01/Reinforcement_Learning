

class LinearConfig:

    num_episodes = int(1e3)
    learning_start = 5
    learning_freq = 4
    max_time_steps_update_epsilon = int(1e6)
    minibatch_size = 5
    replay_buffer_size = int(1e6)
    begin_epsilon = 1
    end_epsilon = 0.1
    gamma = 1
    target_weight_update_freq = 1               # change to 4 or 5 later and see if it makes a difference - why do we not update the weights with every experience? Why do we not always have this as 1?
    frame_stack_size = 4


