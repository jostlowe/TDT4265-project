import gym, random
from collections import deque, namedtuple

from keras import  models, layers, optimizers
import numpy as np
from copy import deepcopy
import time as timer

import matplotlib.pyplot as plt

MAX_EPISODES = 10000

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
        #discount rate
        self.gamma = 0.95
        #exploration rate
        self.epsilon_max = 1.0
        self.epsilon_min = 0.001
        self.epsilon = self.epsilon_max
        self.epsilon_linear_decay = (self.epsilon_min-self.epsilon_max)/(MAX_EPISODES-EPSILON_LINEAR_START)
        self.epsilon_step_decay = self.epsilon_max/EPSILON_STEP_INTERVALS
        self.epsilon_exp_decay = (self.epsilon_min/self.epsilon_max)**((MAX_EPISODES-EPSILON_EXP_START)**(-1))
        #Learning rate
        self.learning_rate_max = 10**(-4)
        self.learning_rate_min = 10**(-6)
        self.learning_rate = self.learning_rate_max
        self.learning_rate_linear_decay = (self.learning_rate_min-self.learning_rate_max)/(MAX_EPISODES-LEARNING_RATE_LINEAR_START)
        self.learning_rate_step_decay = self.learning_rate_max/LEARNING_RATE_STEP_INTERVALS
        self.learning_rate_exp_decay = (self.learning_rate_min/self.learning_rate)**((MAX_EPISODES-LEARNING_RATE_EXP_START)**(-1))
        self.learning_rate_step_exp_decay = (self.learning_rate_min/self.learning_rate)**((MAX_EPISODES-(MAX_EPISODES/LEARNING_RATE_STEP_INTERVALS*LEARNING_RATE_STEP_EXP_START))**(-1))
        self.learning_rate_update = False

        self.model = self._make_model()

    def _make_model(self):
        model = models.Sequential()
        model.add(layers.Dense(50, input_dim=self.num_states, activation='relu'))
        model.add(layers.Dense(50, activation='relu'))
        model.add(layers.Dense(50, activation='relu')) #Added extra
        #model.add(layers.Dense(50, activation='relu')) #Added extra #2
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
            return np.argmax(prediction[0])

    def replay(self, batch_size, episode):
        batch = self.memory.sample(batch_size)

        for state, action, reward, next_state, done in batch:
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(self.model.predict(next_state.fetch())[0])
            target_f = self.model.predict(state.fetch())
            target_f[0][action] = target
            self.model.fit(state.fetch(), target_f, epochs=1, verbose=0)

        if (EPSILON_DECAY_MODE == 'linear'):
            if (episode > EPSILON_LINEAR_START):
                self.epsilon += self.epsilon_linear_decay

        if (EPSILON_DECAY_MODE == 'step') and (episode > 0):
            if (episode % (MAX_EPISODES/EPSILON_STEP_INTERVALS) == 0):
                self.epsilon -= self.epsilon_step_decay

        if (EPSILON_DECAY_MODE == 'exponential'):
            if (episode > EPSILON_EXP_START):
                self.epsilon *= self.epsilon_exp_decay

        if (LEARNING_RATE_DECAY_MODE == 'linear'):
            if (episode > LEARNING_RATE_LINEAR_START):
                self.learning_rate += self.learning_rate_linear_decay

        if (LEARNING_RATE_DECAY_MODE == 'step') and (episode > 0):
            if (episode % (MAX_EPISODES/LEARNING_RATE_STEP_INTERVALS) == 0):
                self.learning_rate -= self.learning_rate_step_decay

        if (LEARNING_RATE_DECAY_MODE == 'exponential'):
            if (episode > LEARNING_RATE_EXP_START):
                self.learning_rate *= self.learning_rate_exp_decay

        if (LEARNING_RATE_DECAY_MODE == 'step-exponential'):
            if (episode % (MAX_EPISODES/LEARNING_RATE_STEP_INTERVALS) == 0) and (episode > 0):
                self.learning_rate = self.learning_rate_max
                self.learning_rate_update = False

            if (episode % (MAX_EPISODES/LEARNING_RATE_STEP_INTERVALS) > MAX_EPISODES/LEARNING_RATE_STEP_INTERVALS*LEARNING_RATE_STEP_EXP_START) or (self.learning_rate_update == True):
                print("ding")
                self.learning_rate_update = True
                self.learning_rate *= self.learning_rate_step_exp_decay

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


if __name__ == "__main__":
    cartpole = gym.make('CartPole-v0')
    num_states = cartpole.observation_space.shape[0]
    num_actions = cartpole.action_space.n
    num_frames = 3

    agent = DQNAgent(num_states=num_states*num_frames, num_actions=num_actions)
    frame_memory = FrameMemory(length=num_frames)
    prev_scores = deque([], 200)
    scores = []
    averages = []
    learnigrates = []
    explorationrates = []

    for episode in range(MAX_EPISODES):
        initial_state = cartpole.reset()
        frame_memory.reset(initial_state)
        score = 0
        for time in range(500):
            memory = frame_memory.fetch()
            action = agent.act(memory)
            prev_frame_memory = deepcopy(frame_memory)
            next_state, reward, done, _ = cartpole.step(action)

            cart_position = next_state[0]
            cart_speed    = next_state[1]
            pole_pos      = next_state[2]
            pole_speed    = next_state[3]
            reward -= abs(cart_position) + 0.01*abs(cart_speed) + 0.005*abs(pole_pos)

            frame_memory.remember(next_state)
            score += reward
            agent.remember((prev_frame_memory, action, reward, deepcopy(frame_memory), done))
            if done:
                print("episode: %i/%i, score = %f\t learning = %f\t e = %f\t" % (episode, MAX_EPISODES, score, agent.learning_rate, agent.epsilon), end=' \t')
                break
        agent.replay(32, episode)
        prev_scores.append(score)
        scores.append(score)
        average = np.mean(list(prev_scores)).round(1)
        averages.append(average)
        learnigrates.append(agent.learning_rate)
        explorationrates.append(agent.epsilon)
        print("avg:", average)

    plt.plot(scores, 'r', averages, 'b')
    plt.show()
    plt.plot(learnigrates)
    plt.show()
    plt.plot(explorationrates)
    plt.show()
    agent.model.save('cartpole.h5')
