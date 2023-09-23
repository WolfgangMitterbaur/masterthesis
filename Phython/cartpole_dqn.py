"""
Created on Tue Mar 16 20:39:44 2023

@author: Wolfgang Mitterbaur

file name cartpole2.py

Deep Q-learning DQN to learn playing cart-and-pole
"""
import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

class DQNAgent:
    '''
    class DQN agent
    '''
    def __init__(self, state_size, action_size):
        '''
        class constructur
        state_size: the length of the state space of the problem
        action_size: the length of the action space of the problem
        '''
        
        self.render = False                         # render shows the image of the cart pole
        self.load_model = True                      # loads the stored weights of the neural network

        # definition of state and action space length
        self.state_size = state_size                # size of the state of the problem
        self.action_size = action_size              # size of the action of the problem

        # DQN hyperparameter
        self.discount_factor = 0.99                 # discount factor for learning algorithm
        self.learning_rate = 0.001                  # learning rate for the adam optimizer
        self.epsilon = 1.0                          # start value of epsilon for greedy methode
        self.epsilon_decay = 0.0003                 # reduction of epsilon each episode
        self.epsilon_min = 0.01                     # minimum value for epsilon
        self.batch_size = 64                        # the size of a batch for learning
        self.train_start = 1000                     # start the training, if 1.000 data are in the relay buffer

        # memory relay buffer with a maximum size of 2.000
        self.memory = deque(maxlen=2000)

        # create the model and the target model
        self.model = self.build_model()             # training model
        self.target_model = self.build_model()      # target model

        # update the target model
        self.update_target_model()

        # load a already trained and stored model
        if self.load_model:
            self.model.load_weights("./cartpole_dqn.h5")

    # create the neural network with input, hidden and output layer
    def build_model(self):
        '''
        public method to bild then neural network model
        '''
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        
        return model

    # update the target model with the weights of the model
    def update_target_model(self):
        '''
        public method to to update the target model
        '''
        self.target_model.set_weights(self.model.get_weights())

    # fetch the action according the epsion greedy method
    def get_action(self, state):
        '''
        public method to to get the next action
        '''
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state, verbose = 0)
            return np.argmax(q_value[0])

    # store the sample (s, a, r, s´)
    def append_sample(self, state, action, reward, next_state, done):
        '''
        public method to to store a additional sample
        state: current state
        action: current action
        reward: current reward
        next_state: next state
        done: episode finished
        '''
        self.memory.append((state, action, reward, next_state, done))

    # train the model with a random sample of the relay buffer
    def train_model(self):
        '''
        public method to to train the model
        '''
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon - self.epsilon_decay

        # create a random minibatch from the relay buffer
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        # create mini-batch for training
        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]            # state
            actions.append(mini_batch[i][1])        # action
            rewards.append(mini_batch[i][2])        # reward
            next_states[i] = mini_batch[i][3]       # next state
            dones.append(mini_batch[i][4])

        # predict a target from the model
        target = self.model.predict(states, verbose = 0)        
        # predict a target value from the target model
        target_val = self.target_model.predict(next_states, verbose = 0)

        # update the target with the bellman optimization equation
        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        self.model.fit(states, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)
         
'''
def main():         
'''
'''
main function
'''  
# environmen cartpole
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# DQN agent is generated
agent = DQNAgent(state_size, action_size)

# the array for the scores and episodes    
scores, episodes = [], []

# number of episodes
max_episodes = 100                          

# loop over all episodes
for e in range(max_episodes):
    
    done = False
    score = 0
    # initialize environment
    state = env.reset()
    state = np.reshape(state, [1, state_size])


    while not done:
        
        if agent.render:
            env.render()

        # fetch the action from the agent
        action = agent.get_action(state)
        # perform the selected action on the environmen
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        
        # reward = reward if not done or score == 499 else -100
        # reward = -100 if the episode ends midway
        if not done or score == 499:
            reward = reward
        else:
            reward = -100                   # done, but not reached 499          
                
        # save the sample (s, a, r, s´)
        agent.append_sample(state, action, reward, next_state, done)
        
        # if the length of the memory is long enough, train the model
        if len(agent.memory) >= agent.train_start:
            agent.train_model()

        score += reward
        state = next_state

        if done:
            # update the target model
            agent.update_target_model()

            # score = score if score == 500 else score + 100
            # output learning results for each episode
            if score == 500:
                score = score                   # reached the maximum of 500
            else:
                score = score + 100             # done, but not reached 500
            
            # output the results of each episode
            scores.append(score)
            episodes.append(e)
            pylab.plot(episodes, scores, 'b')
            pylab.savefig("./cartpole_dqn.png")
            print("episode:", e, "  score:", score, "  memory length:",
                  len(agent.memory), "  epsilon:", agent.epsilon)

            # stop the training, if the mean scores of the last 10 episodes is create than 490
            #if np.mean(scores[-min(10, len(scores)):]) > 490:
            #    agent.model.save_weights("./cartpole_dqn.h5")
            #    sys.exit()
    
sys.exit()
    
'''
main
'''

'''
if __name__ == '__main__':
    main()  
'''



