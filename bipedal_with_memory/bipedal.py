import random
from collections import deque, namedtuple

from keras import  models, layers, optimizers
import numpy as np
from scipy.stats import linregress
from copy import deepcopy

import gym
#from gym_local.envs.box2d import bipedal_walker

MAX_EPISODES = 250000

class ExplorationRate:
    def __init__(self):
        self.max = 1.0
        self.min = 0.001
        self.epsilon = self.max
        # Choose between 'none' 'linear', 'steps' and 'exponential'
        self.decay_mode = 'steps'
        self.decay_functions = {
            'none' : self.no_decay,
            'linear' : self.linear_decay,
            'steps' : self.step_decay,
            'exponential' : self.exp_decay,
        }
        self.decay = self.decay_functions.get(self.decay_mode)

    def get(self):
        return self.epsilon

    def no_decay(self, episode):
        return self.epsilon

    def linear_decay(self, episode):
        start = MAX_EPISODES/10

        if (episode > start):
            decay = (self.min-self.max)/(MAX_EPISODES - start)
            self.epsilon += decay

        return self.epsilon

    def step_decay(self, episode):
        steps = 10

        if (episode > 0) and (episode % (MAX_EPISODES//steps) == 0):
            decay = -(self.min + self.max)/(steps)
            self.epsilon += decay

        return self.epsilon

    def exp_decay(self, episode):
        start = MAX_EPISODES/10

        if (episode > start):
            decay = (self.min/self.max)**((MAX_EPISODES-start)**(-1))
            self.epsilon *= decay

        return self.epsilon

class LearningRate:
    def __init__(self):
        self.max = 10**(-4)
        self.min = 10**(-6)
        self.alpha = self.max
        # Choose between 'none' 'linear', 'steps', 'exponential' and 'step-exponential'
        self.decay_mode = 'none'
        self.decay_functions = {
            'none' : self.no_decay,
            'linear' : self.linear_decay,
            'steps' : self.step_decay,
            'exponential' : self.exp_decay,
            'step-exponential' : self.step_exp_decay,
        }
        self.decay = self.decay_functions.get(self.decay_mode)

    def get(self):
        return self.alpha

    def no_decay(self, episode):
        return self.alpha

    def linear_decay(self, episode):
        start = MAX_EPISODES/2

        if (episode > start):
            decay = (self.min-self.max)/(MAX_EPISODES-start)
            self.alpha += decay

        return self.alpha

    def step_decay(self, episode):
        steps = 10

        if (episode > 0) and (episode % (MAX_EPISODES//steps) == 0):
            decay = -(self.min+self.max)/steps
            self.alpha += decay

        return self.alpha

    def exp_decay(self, episode):
        start = 8*MAX_EPISODES/10

        if (episode > start):
            decay = (self.min/self.max)**((MAX_EPISODES-start)**(-1))
            self.alpha *= decay

        return self.alpha

    def step_exp_decay(self, episode):
        steps = 10
        start = 1/3 #Starting point of exponential decay in each step

        if (episode % (MAX_EPISODES//steps) == 0):
            self.alpha = self.max

        if (episode % (MAX_EPISODES//steps) > MAX_EPISODES//steps*start):
            decay = (self.min/self.max)**((MAX_EPISODES-(MAX_EPISODES/steps*start))**(-1))
            self.alpha *= decay

        return self.alpha


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], capacity)

    # transitions on format (state, action, reward, next_state, done)
    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.choices(self.memory, k=batch_size)


class DQNAgent:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        self.memory = ReplayMemory(capacity=MAX_EPISODES//50)
        #discount rate
        self.gamma = 0.98
        #exploration rate
        self.exploration_rate = ExplorationRate()
        self.epsilon = self.exploration_rate.get()
        #Learning rate
        self.learning_rate = LearningRate()
        self.alpha = self.learning_rate.get()

        self.model = self._make_model()


    def _make_model(self):
        model = models.Sequential()
        model.add(layers.Dense(300, input_dim=self.num_states, activation='relu'))
        model.add(layers.Dense(300, activation='relu'))
        model.add(layers.Dense(self.num_actions, activation='linear'))
        model.compile(loss='mse', optimizer=optimizers.Adam(lr=self.alpha))
        return model

    def remember(self, transition):
        self.memory.push(transition)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # return random action
            return random.randrange(self.num_actions)
        else:
            prediction = self.model.predict(state)
            # returns action with highest q-value
            return np.argmax(prediction)

    def replay(self, batch_size, episode):
        batch = self.memory.sample(batch_size)
        for frame_memory, action, reward, next_frame_memory, done in batch:
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(self.model.predict(next_frame_memory.fetch())[0])
            target_f = self.model.predict(frame_memory.fetch())
            target_f[0][action] = target
            self.model.fit(frame_memory.fetch(), target_f, epochs=1, verbose=0)

        self.epsilon = self.exploration_rate.decay(episode)

        self.alpha = self.learning_rate.decay(episode)


def calculate_action(action_number):
    action_space = {
        1: [-1,0,0,0],
        2: [1,0,0,0],
        3: [0,-1,0,0],
        4: [0,1,0,0],
        5: [0,0,-1,0],
        6: [0,0,1,0],
        7: [0,0,0,-1],
        0: [0,0,0,1],
    }
    return action_space[action_number]

def calculate_slope(scores):
    x = [i for i in range(len(scores))]
    y = list(scores)

    slope, _,_,_,_= linregress(x,y)
    return slope


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




if __name__ == "__main__":
    prev_scores = deque([], MAX_EPISODES//50)
    bipedal = gym.make('BipedalWalker-v2')
    num_states = bipedal.observation_space.shape[0]
    num_actions = bipedal.action_space.shape[0]
    num_frames = 3
    LOAD = False

    frame_memory = FrameMemory(length=num_frames)
    agent = DQNAgent(num_states=num_states*num_frames, num_actions=8)
    if LOAD:
        agent.model = models.load_model('bipedal.h5')

    for episode in range(MAX_EPISODES):
        initial_state = bipedal.reset()
        frame_memory.reset(initial_state)
        score = 0

        for time in range(500):
            action_number = agent.act(frame_memory.fetch())
            action = calculate_action(action_number)
            prev_frame_memory = deepcopy(frame_memory)
            next_state, reward, done, _ = bipedal.step(action)

            if (reward == -100):
                reward = -50

            frame_memory.remember(next_state)
            agent.remember((prev_frame_memory, action, reward, deepcopy(frame_memory), done))
            score += reward
            if done:
                break
        prev_scores.append(score)
        slope = calculate_slope(prev_scores)
        #print("episode: %i/%i -> %i  \t slope: %f\t epsilon: %f" % (episode, MAX_EPISODES, score, slope, agent.epsilon))
        with open('data.csv', 'a') as csv_file:
            csv_file.write("%i, %i, %f, %f\n" % (episode, score, slope, agent.epsilon))

        agent.replay(32, episode)

        if episode % 100 == 0:
            print("checkpoint")
            agent.model.save('bipedal.h5')
