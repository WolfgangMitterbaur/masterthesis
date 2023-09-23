"""
Created on Tue Mar 16 20:39:44 2023

@author: Wolfgang Mitterbaur

file name cartpole4.py

Advantage actor-critic agent to learn playing cart-and-pole
"""

import sys
import gym
import pylab
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

class A2CAgent:
    '''
    class A2C agent
    '''
    def __init__(self, state_size, action_size):
        '''
        class constructur
        state_size: the length of the state space of the problem
        action_size: the length of the action space of the problem
        '''
        self.render = False                                 # render shows the image of the cart pole
        self.load_model = False                             # loads the stored weights of the neural network
        
        # definition of state and action space length
        
        self.state_size = state_size                        # size of the state of the problem
        self.action_size = action_size                      # size of the action of the problem
        self.value_size = 1

        # policy gradient hyperparameters
        self.discount_factor = 0.99                         # discount factor for learning algorithm
        self.actor_lr = 0.001                               # learning rate for the actor optimizer
        self.critic_lr = 0.005                              # learning rate for the critic optimizer

        # call the building blocks
        self.actor = self.build_actor()
        self.critic = self.build_critic()

        # load a model
        if self.load_model:
            self.actor.load_weights("./cartpole_actor.h5")
            self.critic.load_weights("./cartpole_critic.h5")
    
    # create the neural network for the approximation of the actor and critic values (policy and value for the model)
    
    def build_actor(self):
        '''
        public method to bild the actor neural network
        actor module: input of states and outputs the probability of an action (softmax)
        '''
        actor = Sequential()                                            
        actor.add(Dense(24 , input_dim = self.state_size, activation= 'relu', kernel_initializer= 'he_uniform'))
        actor.add(Dense(self.action_size, activation='softmax',kernel_initializer='he_uniform'))
        actor.summary()
        actor.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = self.actor_lr))
        return actor
    
    def build_critic(self):
        '''
        public method to bild the critic neural network
        critic module: Input is also state but the output is also state (linear)
        '''
        critic = Sequential()
        critic.add(Dense(24, input_dim = self.state_size, activation= 'relu', kernel_initializer= 'he_uniform'))
        critic.add(Dense(self.value_size,activation= 'linear', kernel_initializer= 'he_uniform'))
        critic.summary()
        critic.compile(loss = 'mse', optimizer= Adam(lr=self.critic_lr))        # Loss is MSE since we want to give out a value and not a probability.
        return critic
    
    def get_action(self,state):
        '''
        public method to fetch the next action from the actor
        function on how the agent will pick the next action and policy based on stochastics(probability)
        '''
        policy = self.actor.predict(state, batch_size = 1, verbose = 0).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]
    
    
    def train_model(self, state, action, reward, next_state, done):
        '''
        public method to train the model: update for the network policy
        '''
        target = np.zeros((1, self.value_size))                                 # initialize the policy targets matrix
        advantages = np.zeros((1, self.action_size))                            # initialize the advantages matrix

        value = self.critic.predict(state, verbose = 0)[0]                      # get value for this state
        next_value = self.critic.predict(next_state, verbose = 0)[0]            # get value for the next state

        # update the advantages and value tables if done
        if done:
            advantages[0][action] = reward - value                              # basically, what do we gain by choosing the action, will it improve or worsen the advantage
            target[0][0] = reward                                               # fill in the target value to see if we can still improve it in the policy making
        else:
            advantages[0][action] = reward + self.discount_factor*(next_value) - value  # if not yet done, then simply update for the current step
            target[0][0] = reward + self.discount_factor*next_value
        
        # update the weights of the actor and the critic
        self.actor.fit(state, advantages, epochs = 1, verbose = 0)
        self.critic.fit(state, target, epochs = 1, verbose = 0)



def main():         
    '''
    main function
    ''' 
    
    # create an environment
    env = gym.make('CartPole-v1')
    
    # get the action and state sizes
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    #action_size = env.action_space.shape[0]
    
    # make the agent by calling the function earlier
    agent = A2CAgent(state_size,action_size)
    
    # initialize our scores and episodes list
    scores, episodes = [], []

    # number of episodes
    max_episodes = 100
    
    # create the training loop
    for e in range(max_episodes):
        
        done = False                                            # done-flag if a episode is finished
        score = 0                                               # the score of all episodes
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            
            # render the image
            if agent.render:
                env.render()
            
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            
            # give immediate penalty for an action that terminates the episode immediately, since we want to maximize the time
            # note that the max for the cartpole is 499 and it will reset, 
            # otherwise we keep the current score if it is not yet done, and if it ended we give a -100 reward
            
            # reward = reward if not done or score == 499 else -100
            if not done or score == 499:
                reward = reward
            else:
                reward = -100        # done, but not reached 499 
            
            # train the model based on the results of the action taken
            agent.train_model(state, action, reward, next_state, done)
            score += reward
            state = next_state

            if done:
                #score = score if score == 500.0 else score +100
                if score == 500.0:
                    score = score                # reached the maximum of 500
                else:
                    score = score + 100          # done, but not reached 500
            
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes,scores,'b')
                pylab.savefig("./A2C-CartPole.png")
                #if np.mean(scores[-min(10, len(scores)):]) > 490:
                #    sys.exit()
        
        # every 50 episodes save the model weiths and print the results
        if e % 50 == 0:
        #    agent.actor.save_weights("./cartpole_actor.h5")
        #    agent.critic.save_weights("./cartpole_critic.h5")
            print("episode: {} score: {}".format(e,score))
            
            
'''
main
'''
if __name__ == '__main__':
    main()  


