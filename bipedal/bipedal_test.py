import gym

from keras import models
from cartpole.cartpole import DQNAgent

import numpy as np

cartpole = gym.make('BipedalWalker-v2')
num_states = cartpole.observation_space.shape[0]
num_actions = cartpole.action_space.shape[0]

agent = DQNAgent(num_states, num_actions)
agent.model = models.load_model('bipedal.h5')
agent.epsilon = agent.epsilon_min

def calculate_action(action_number):
    themap = {
        1: [-1,0,0,0],
        2: [1,0,0,0],
        3: [0,-1,0,0],
        4: [0,1,0,0],
        5: [0,0,-1,0],
        6: [0,0,1,0],
        7: [0,0,0,-1],
        0: [0,0,0,1],
    }
    return themap[action_number]


while True:
    done = False
    state = cartpole.reset()
    state = np.reshape(state, [1, num_states])
    cartpole.render()
    while not done:
        action_number = agent.act(state)
        action = calculate_action(action_number)
        state, reward, done, info = cartpole.step(action)
        state = np.reshape(state, [1, num_states])
        cartpole.render()
        #time.sleep()
