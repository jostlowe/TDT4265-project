import gym

from keras import models
from bipedal import DQNAgent
from collections import deque
import itertools
import numpy as np

bipedal = gym.make('BipedalWalker-v2')
num_states = bipedal.observation_space.shape[0]
num_actions = bipedal.action_space.shape[0]

agent = DQNAgent(num_states*10, 81)
agent.model = models.load_model('bipedal.h5')
agent.epsilon = agent.epsilon_min


def calculate_action(action_number):
    temp = list(itertools.product([-1, 0, 1], repeat=4))
    the_map = {}
    for i in range(len(temp)):
        the_map[i] = temp[i]
    return list(the_map[action_number])


class FrameMemory:
    def __init__(self, length):
        self.length = length
        self.queue = deque([], length)

    def reset(self, state):
        for _ in range(self.length):
            self.queue.append(state)

    def remember(self, state):
        self.queue.append(state)

    def fetch(self):
        return np.reshape(list(self.queue), [1, len(self.queue[0])*self.length])


frame_memory = FrameMemory(10)

while True:
    done = False
    state = bipedal.reset()
    frame_memory.reset(state)
    bipedal.render()
    while not done:
        action_number = agent.act(frame_memory.fetch())
        action = calculate_action(action_number)
        state, reward, done, info = bipedal.step(action)
        frame_memory.remember(state)
        bipedal.render()
        #time.sleep()
