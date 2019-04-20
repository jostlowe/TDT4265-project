import gym, random
from collections import deque, namedtuple

from keras import  models, layers, optimizers
import numpy as np
from scipy.stats import linregress
import time as timer

MAX_EPISODES = 100000

#Chose modes
#Choose vetween 'linear', 'step', 'exponential', or 'exponential-step' (only for learning)
EPSILON_DECAY_MODE = 'exponential'
LEARNING_RATE_DECAY_MODE = 'exponential'

#Constants for linear decay
EPSILON_LINEAR_START = MAX_EPISODES/10
LEARNING_RATE_LINEAR_START = MAX_EPISODES/2

#Constants for step-wise decay
EPSILON_STEP_INTERVALS = 10
LEARNING_RATE_STEP_INTERVALS = 10

#Constants for exponential decay
EPSILON_EXP_START = MAX_EPISODES/10
LEARNING_RATE_EXP_START = 8*MAX_EPISODES/10

#Constants for step-wise exponential decay (uses LEARNING_STEP_INTERVALS)
LEARNING_RATE_STEP_EXP_START = 1/3


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
        self.memory = ReplayMemory(capacity=2000)
        self.gamma = 0.98    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 1-5/MAX_EPISODES
        self.learning_rate = 0.001
        self.model = self._make_model()


    def _make_model(self):
        model = models.Sequential()
        model.add(layers.Dense(400, input_dim=self.num_states, activation='relu'))
        model.add(layers.Dense(300, activation='relu'))
        model.add(layers.Dense(300, activation='relu'))
        model.add(layers.Dense(self.num_actions, activation='linear'))
        model.compile(loss='mse', optimizer=optimizers.Adam(lr=self.learning_rate))
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
        for state, action, reward, next_state, done in batch:
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if (self.epsilon > self.epsilon_min) and (episode > MAX_EPISODES//10):
            self.epsilon *= self.epsilon_decay


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

def calculate_slope(scores):
    x = [i for i in range(len(scores))]
    y = list(scores)

    slope, _,_,_,_= linregress(x,y)
    return slope




if __name__ == "__main__":
    prev_scores = deque([0,0], MAX_EPISODES//50)
    bipedal = gym.make('BipedalWalker-v2')
    num_states = bipedal.observation_space.shape[0]
    num_actions = bipedal.action_space.shape[0]

    agent = DQNAgent(num_states=num_states, num_actions=8)

    for episode in range(MAX_EPISODES):
        state = bipedal.reset()
        state = np.reshape(state, [1, num_states])
        score = 0

        for time in range(200):
            action_number = agent.act(state)
            action = calculate_action(action_number)
            next_state, reward, done, _ = bipedal.step(action)
            next_state = np.reshape(next_state, [1, num_states])
            agent.remember((state, action, reward, next_state, done))
            state = next_state
            score += reward
            if done:
                break
        avg = np.average(prev_scores)
        print("episode: %i/%i -> %i  \t avg: %f\t epsilon: %f" % (episode, MAX_EPISODES, score, avg, agent.epsilon))
        with open('data.csv', 'a') as csv_file:
            csv_file.write("%i, %i, %f, %f\n" % (MAX_EPISODES, score, avg, agent.epsilon))

        agent.replay(32, episode)
        prev_scores.append(score)
        if episode % 100 == 0:
            print("checkpoint")
            agent.model.save('bipedal.h5')
