import gym, random
from collections import deque, namedtuple

from keras import  models, layers, optimizers
import numpy as np
from scipy.stats import linregress
from copy import deepcopy
import time as timer

MAX_EPISODES = 1000

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
        self.memory = ReplayMemory(capacity=500)
        self.gamma = 0.98    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 1-5/MAX_EPISODES
        self.learning_rate = 1/MAX_EPISODES
        self.model = self._make_model()


    def _make_model(self):
        model = models.Sequential()
        model.add(layers.Dense(300, input_dim=self.num_states, activation='relu'))
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
        for frame_memory, action, reward, next_frame_memory, done in batch:
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(self.model.predict(next_frame_memory.fetch())[0])
            target_f = self.model.predict(frame_memory.fetch())
            target_f[0][action] = target
            self.model.fit(frame_memory.fetch(), target_f, epochs=1, verbose=0)
        if (self.epsilon > self.epsilon_min) and (episode > MAX_EPISODES//10):
            self.epsilon *= self.epsilon_decay


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
    num_frames = 10
    LOAD = True

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
            frame_memory.remember(next_state)
            agent.remember((prev_frame_memory, action, reward, deepcopy(frame_memory), done))
            score += reward
            if done:
                break
        prev_scores.append(score)
        slope = calculate_slope(prev_scores)
        print("episode: %i/%i -> %i  \t slope: %f\t epsilon: %f" % (episode, MAX_EPISODES, score, slope, agent.epsilon))
        with open('data.csv', 'a') as csv_file:
            csv_file.write("%i, %i, %f, %f\n" % (episode, score, slope, agent.epsilon))

        agent.replay(8, episode)

        if episode % 100 == 0:
            print("checkpoint")
            agent.model.save('bipedal.h5')
