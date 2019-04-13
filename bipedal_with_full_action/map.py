import itertools

def calculate_action(action_number):
    temp = list(itertools.product([-1, 0, 1], repeat=4))
    action_space = {}
    for i in range(len(temp)):
        action_space[i] = temp[i]
    return action_space[action_number]
