import numpy as np

rows, cols = 32, 17

# Fill in track with 1 where it is a valid location for car to be in
track = np.zeros((rows,cols))
track[0:6, 10::] = 1
track[0:7, 9] = 1
track[0::, 3:9] = 1
track[1:29, 2] = 1
track[3:22, 1] = 1
track[4:14, 0] = 1

# finish line
# track[0:6, cols-1] = 10
# starting_line = (rows-1, slice(3,9))


def get_random_start_col():
    start_col_begin = 3
    start_col_end = 8
    new_col = np.random.randint(start_col_begin, start_col_end + 1)
    return new_col

actions = []
a_set = [0, -1, 1]
for horizontal_action in a_set:
    for vertical_action in a_set:
        actions.append([horizontal_action, vertical_action])



MAX_VELOCITY = 5

'''
Simply using algebra to see if line segment created by path intersects
the finish line (at col = 16) between rows 0 and 5 (inclusive)
'''
def crossed_finishline(row, col, new_row, new_col):

    FINISH_LINE_COL = 16

    x = col
    y = row

    x_new = new_col
    y_new = new_row

    m = ((y_new - y) / (x_new - x))
    b = y_new - (m*x_new)
    y_intersect = (m*FINISH_LINE_COL) + b

    if 0 <= y_intersect <= 5:
        return True
    else:
        return False


def crossed_boundary(new_row, new_col):

    if (((new_row<0) or (rows<=new_row)) or ((new_col<0) or (cols<=new_col))):
        return True

    if track[new_row][new_col] == 0:
        return True

    return False


def generate_episode(policy):
    vertical_velocity = 0
    horizontal_velocity = 0
    episode = []
    row = rows-1
    col = get_random_start_col()

    while (True):
        reward = -1
        #                 state (t)  action (t)         reward (t+1)
        episode.append( ( (row,col), policy[(row,col)] ,reward ) )

        horizontal_increment = 0
        vertical_increment = 0
        if np.random.binomial(n=1, p=0.1):
            horizontal_increment = 0
            vertical_increment = 0
        else:
            horizontal_increment = policy[(row,col)][0]
            vertical_increment = policy[(row,col)][1]

        vertical_velocity = min(MAX_VELOCITY, vertical_velocity + vertical_increment)
        horizontal_velocity = min(MAX_VELOCITY, horizontal_velocity + horizontal_increment)

        new_row = row + vertical_velocity
        new_col = col + horizontal_velocity

        if crossed_finishline(row, col, new_row, new_col):
            break

        if crossed_boundary(new_row, new_col):
            row = rows - 1
            col = get_random_start_col()
        else:
            row = new_row
            col = new_col


    return episode







