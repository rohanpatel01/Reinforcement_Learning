


class EpsilonScheduler:

    def __init__(self, begin_epsilon, end_epsilon, max_time_steps_update_epsilon):
        self.begin_epsilon = begin_epsilon
        self.end_epsilon = end_epsilon
        self.max_time_steps_update_epsilon = max_time_steps_update_epsilon
        self.epsilon = begin_epsilon

    def get_epsilon(self, time):
        time = max(time, 0)
        if time <= self.max_time_steps_update_epsilon:
            self.epsilon = (((self.begin_epsilon - self.end_epsilon) / -self.max_time_steps_update_epsilon) * time) + self.begin_epsilon
        else:
            self.epsilon = self.end_epsilon

        return self.epsilon


    def reset(self):
        self.epsilon = self.begin_epsilon


if __name__ == '__main__':
    pass
