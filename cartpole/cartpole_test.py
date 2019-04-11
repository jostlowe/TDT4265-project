import gym, time

from keras import models
from cartpole import DQNAgent

import numpy as np

cartpole = gym.make('CartPole-v0')
num_states = cartpole.observation_space.shape[0]
num_actions = cartpole.action_space.n
#cartpole = gym.wrappers.Monitor(cartpole, './video/', force=True)

agent = DQNAgent(num_states, num_actions)
agent.model = models.load_model('cartpole.h5')
agent.epsilon = agent.epsilon_min


while True:
    done = False
    state = cartpole.reset()
    state = np.reshape(state, [1, 4])
    cartpole.render()
    while not done:
        action = agent.act(state)
        state, reward, done, info = cartpole.step(action)
        state = np.reshape(state, [1, 4])
        cartpole.render()
        time.sleep(.05)
