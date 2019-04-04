import gym, time, random
from collections import deque, namedtuple

from keras import  models, layers, optimizers
import numpy as np



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
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
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

    def replay(self, batch_size):
        batch = self.memory.sample(batch_size)
        for state, action, reward, next_state, done in batch:
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay




cartpole = gym.make('CartPole-v0')
num_states = cartpole.observation_space.shape[0]
num_actions = cartpole.action_space.n
max_episodes = 1000

agent = DQNAgent(num_states=num_states, num_actions=num_actions)
prev_scores = deque([], 20)

for e in range(max_episodes):
    state = cartpole.reset()
    state = np.reshape(state, [1, 4])

    for time in range(500):
        action = agent.act(state)
        next_state, reward, done, _ = cartpole.step(action)
        next_state = np.reshape(next_state, [1, 4])
        agent.remember((state, action, reward, next_state, done))
        state = next_state
        if done:
            print("episode: %i/%i, score = %i" % (e, max_episodes, time), end=' \t')
            break
    agent.replay(32)
    prev_scores.append(time)
    print("avg:", np.mean(list(prev_scores)).round(1))


agent.model.save('cartpole.h5')