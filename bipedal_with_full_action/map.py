import itertools

def calculate_action(action_number):
    temp = list(itertools.product([-1, 0, 1], repeat=4))
    themap = {}
    for i in range(len(temp)):
        themap[i] = temp[i]
    return themap[action_number]