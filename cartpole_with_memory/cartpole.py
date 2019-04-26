import gym, random
from collections import deque, namedtuple

from keras import  models, layers, optimizers
import numpy as np
from copy import deepcopy
import time as timer

import matplotlib.pyplot as plt

MAX_EPISODES = 10000

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
        decay = (self.min-self.max)/(MAX_EPISODES - start)

        if (episode > start):
            self.epsilon += decay

        return self.epsilon

    def step_decay(self, episode):
        steps = 10
        decay = (self.min - self.max)/(steps)

        if (episode > 0) and (episode % (MAX_EPISODES//steps) == 0):
            self.epsilon += decay

        return self.epsilon

    def exp_decay(self, episode):
        start = MAX_EPISODES/10
        decay = (self.min/self.max)**((MAX_EPISODES-start)**(-1))

        if (episode > start):
            self.epsilon *= decay

        return self.epsilon

class LearningRate:
    def __init__(self):
        self.max = 10**(-4)
        self.min = 10**(-6)
        self.alpha = self.max
        # Choose between 'none' 'linear', 'steps' and 'exponential'
        self.decay_mode = 'step-exponential'
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
        decay = (self.min-self.max)/(MAX_EPISODES-start)

        if (episode > start):
                self.alpha += decay

        return self.alpha

    def step_decay(self, episode):
        steps = 10
        decay = (self.min-self.max)/steps

        if (episode > 0) and (episode % (MAX_EPISODES//steps) == 0):
            self.alpha += decay

        return self.alpha

    def exp_decay(self, episode):
        start = 8*MAX_EPISODES/10
        decay = (self.min/self.max)**((MAX_EPISODES-start)**(-1))

        if (episode > start):
            self.alpha *= decay

        return self.alpha

    def step_exp_decay(self, episode):
        steps = 10
        start = 1/3 #Starting point of exponential decay in each step
        update = False

        if (episode % (MAX_EPISODES//steps) == 0):
            self.alpha = self.max
            self.update = False

        if (episode % (MAX_EPISODES//steps) > MAX_EPISODES//steps*start) or (update == True):
            update = True
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
        self.gamma = 0.95
        #exploration rate
        self.exploration_rate = ExplorationRate()
        self.epsilon = self.exploration_rate.get()
        #Learning rate
        self.learning_rate = LearningRate()
        self.alpha = self.learning_rate.get()

        self.model = self._make_model()

    def _make_model(self):
        model = models.Sequential()
        model.add(layers.Dense(50, input_dim=self.num_states, activation='relu'))
        model.add(layers.Dense(50, activation='relu'))
        model.add(layers.Dense(50, activation='relu')) #Added extra
        #model.add(layers.Dense(50, activation='relu')) #Added extra #2
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

        self.epsilon = self.exploration_rate.decay(episode)

        self.alpha = self.learning_rate.decay(episode)


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
                print("episode: %i/%i, score = %f\t learning = %f\t e = %f\t" % (episode, MAX_EPISODES, score, agent.alpha, agent.epsilon), end=' \t')
                break
        agent.replay(32, episode)
        prev_scores.append(score)
        scores.append(score)
        average = np.mean(list(prev_scores)).round(1)
        averages.append(average)
        learnigrates.append(agent.alpha)
        explorationrates.append(agent.epsilon)
        print("avg:", average)

    plt.plot(scores, 'r', averages, 'b')
    plt.show()
    plt.plot(learnigrates)
    plt.show()
    plt.plot(explorationrates)
    plt.show()
    agent.model.save('cartpole.h5')
