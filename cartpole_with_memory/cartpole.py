import gym, random
from collections import deque, namedtuple

from keras import  models, layers, optimizers
import numpy as np
from copy import deepcopy
import time as timer

import matplotlib.pyplot as plt


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
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.0001
        self.model = self._make_model()


    def _make_model(self):
        model = models.Sequential()
        model.add(layers.Dense(24, input_dim=self.num_states, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
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
        if (self.epsilon > self.epsilon_min):
            self.epsilon *= self.epsilon_decay


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
        return np.reshape(self.queue[0], [1, 2])


if __name__ == "__main__":
    cartpole = gym.make('CartPole-v0')
    num_states = cartpole.observation_space.shape[0]
    num_actions = cartpole.action_space.n
    num_frames = 3
    max_episodes = 10000

    agent = DQNAgent(num_states=num_states*num_frames, num_actions=num_actions)
    frame_memory = FrameMemory(length=num_frames)
    prev_scores = deque([], 200)
    scores = []
    averages = []

    for e in range(max_episodes):
        initial_state = cartpole.reset()
        frame_memory.reset(initial_state)
        score = 0
        for time in range(500):
            action = agent.act(frame_memory.fetch())
            prev_frame_memory = deepcopy(frame_memory)
            next_state, reward, done, _ = cartpole.step(action)
            frame_memory.remember(next_state)
            score += reward
            agent.remember((prev_frame_memory, action, reward, frame_memory, done))
            if done:
                print("episode: %i/%i, score = %f e=%f" % (e, max_episodes, score, agent.epsilon), end=' \t')
                break
        agent.replay(32, e)
        prev_scores.append(score)
        scores.append(score)
        average = np.mean(list(prev_scores)).round(1)
        averages.append(average)
        print("avg:", average)

    plt.plot(scores, 'r', averages, 'b')
    plt.show()
    agent.model.save('cartpole.h5')
