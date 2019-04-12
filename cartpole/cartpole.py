import gym, random
from collections import deque, namedtuple

from keras import  models, layers, optimizers
import numpy as np
import time as timer

import matplotlib.pyplot as plt

MAX_EPISODES = 10000

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
        self.epsilon_decay = 1-5/MAX_EPISODES
        self.learning_rate = 1/MAX_EPISODES
        self.model = self._make_model()


    def _make_model(self):
        model = models.Sequential()
        model.add(layers.Dense(24, input_dim=self.num_states, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(24, activation='relu')) #Added extra
        model.add(layers.Dense(24, activation='relu')) #Added extra #2
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
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if (self.epsilon > self.epsilon_min) and (episode > MAX_EPISODES//10):
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    cartpole = gym.make('CartPole-v0')
    num_states = cartpole.observation_space.shape[0]
    num_actions = cartpole.action_space.n

    agent = DQNAgent(num_states=num_states, num_actions=num_actions)
    prev_scores = deque([], MAX_EPISODES//50)
    scores = []
    averages = []

    for episode in range(MAX_EPISODES):
        state = cartpole.reset()
        state = np.reshape(state, [1, 4])
        score = 0
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = cartpole.step(action)
            cart_position = next_state[0]
            cart_speed    = next_state[1]
            pole_pos      = next_state[2]
            pole_speed    = next_state[3]
            reward -= abs(cart_position) + 0.01*abs(cart_speed) + 0.005*abs(pole_pos)
            score += reward
            next_state = np.reshape(next_state, [1, 4])
            agent.remember((state, action, reward, next_state, done))
            state = next_state
            if done:
                print("episode: %i/%i, e = %f, score = %f" % (episode, MAX_EPISODES, agent.epsilon, score), end=' \t')
                break
        agent.replay(32, episode)
        prev_scores.append(score)
        scores.append(score)
        average = np.mean(list(prev_scores)).round(1)
        averages.append(average)
        print("avg:", average)

    plt.plot(scores, 'r', averages, 'b')
    plt.show()
    agent.model.save('cartpole.h5')
