

class LinearConfig:

    learning_freq = 4
    minibatch_size = 10
    replay_buffer_size = int(1e2)
    begin_epsilon = 1
    end_epsilon = 0.1
    max_time_steps_update_epsilon = int(1e6)
    num_episodes = int(1e2)
    gamma = 1
    learning_start = 200
    target_weight_update_freq = 1               # change to 4 or 5 later and see if it makes a difference - why do we not update the weights with every experience? Why do we not always have this as 1?
    frame_stack_size = 4


