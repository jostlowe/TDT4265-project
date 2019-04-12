import gym, time
from collections import deque, namedtuple

from keras import models
from cartpole import DQNAgent

import numpy as np

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

    def latest(self):
        return np.reshape(self.queue[self.length], [1, 4])


cartpole = gym.make('CartPole-v0')
num_states = cartpole.observation_space.shape[0]
num_actions = cartpole.action_space.n
num_frames = 3

agent = DQNAgent(num_states=num_states*num_frames, num_actions=num_actions)
frame_memory = FrameMemory(length=num_frames)
agent.model = models.load_model('cartpole.h5')
agent.epsilon = agent.epsilon_min


while True:
    done = False
    initial_state = cartpole.reset()
    frame_memory.reset(initial_state)
    cartpole.render()
    while not done:
        memory = frame_memory.fetch()
        action = agent.act(memory)
        next_state, reward, done, info = cartpole.step(action)
        frame_memory.remember(next_state)
        cartpole.render()
        time.sleep(.05)
