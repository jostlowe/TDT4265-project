import gym

from keras import models
from bipedal import DQNAgent

import numpy as np

bipedal = gym.make('BipedalWalker-v2')
num_states = bipedal.observation_space.shape[0]
num_actions = bipedal.action_space.shape[0]

agent = DQNAgent(num_states, num_actions)
agent.model = models.load_model('bipedal_spaghoot.h5')
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
    state = bipedal.reset()
    state = np.reshape(state, [1, num_states])
    bipedal.render()
    while not done:
        action_number = agent.act(state)
        action = calculate_action(action_number)
        state, reward, done, info = bipedal.step(action)
        state = np.reshape(state, [1, num_states])
        bipedal.render()
        #time.sleep()
