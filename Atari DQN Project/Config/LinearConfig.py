

class LinearConfig:

    learning_freq = 5
    minibatch_size = 10
    replay_buffer_size = 1e6
    begin_epsilon = 1
    end_epsilon = 0.1
    max_time_steps_update_epsilon = 1e6
    num_episodes = 1e3
    gamma = 1