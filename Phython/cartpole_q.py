"""
Created on Tue Mar 16 20:39:44 2023

@author: Wolfgang Mitterbaur

file name cartpole1.py

Q-learning to learn playing cart-and-pole
"""
import gym
import numpy as np 
import matplotlib.pyplot as plt
import time
import pylab

# global variables
data = {'max' : [0], 'avg' : [0]}       # the maximum value and the average value of reward
lastepi = []                            # the last 100 episodes: episodes number
lastscore = []                          # the last 100 episodes: scores

# cart pole environment
env = gym.make('CartPole-v1')
print(env.observation_space.low,"\n",env.observation_space.high)

def Qtable(state_space, action_space, bin_size = 30):
    '''
    definition of Q table
    state_space: the length of the state space
    action_space: the length of action space
    bin_size: the size of each dimension
    '''
    bins = [np.linspace(-4.8, 4.8, bin_size),                       # cart position -4.8 to +4.8
            np.linspace(-4, 4, bin_size),                           # cart velocity -4.0 to +4.0
            np.linspace(-0.418, 0.418, bin_size),                   # pole angle -0.418 to +0.418 rad
            np.linspace(-4, 4, bin_size)]                           # pole velocity -4 to +4
    
    q_table = np.random.uniform(low=-1, high=1, size=([bin_size] * state_space + [action_space]))
    
    return q_table, bins


def Discrete(state, bins):
    '''
    the continuous state is converted into a discrete state,
    which fits the closest state value in the q-table
    bins: defines the discrete values
    state: is the current continous value
    index: is the return value of the best fitting dimension in the q-table.
    '''
    
    index = []
    
    for i in range(len(state)): index.append(np.digitize(state[i], bins[i]) - 1)
    
    return tuple(index)


def Q_learning(q_table, bins, episodes = 5000, gamma = 0.95, lr = 0.1, timestep = 5000, epsilon = 0.2):
    '''
    q-learning algorithm
    q-table: the table with all q-values
    bins: the array with all posible values for each dimension
    episodes: the number of maximum episodes of training
    gamma: the gamma for q-learning
    lr = the learning rate
    timestep = the timestep
    epsilon: refers to the exploration vs. exploitation problem
    '''
    
    render = False    
    rewards = 0                                                     # start reward
    solved = False                                                  # flag of solved state
    steps = 0                                                       # the counter of learning steps
    runs = [0]                                                      # the score of each run
    ep = [i for i in range(0, episodes + 1, timestep)]              # time array: one value for each timestep
    
    # loop for all episodes
    for episode in range(1, episodes + 1):
        
        current_state = Discrete(env.reset(), bins)                 # initial observation
        score = 0                                                   # the start score
        done = False                                                # the done flag
        
        # loop for traiing
        while not done:
            steps += 1                                              # increase the number of steps
            ep_start = time.time()                                  # store the start time stamp
            if render and episode%timestep == 0:
                env.render()
                
            if np.random.uniform(0, 1) < epsilon:                   # calculate a random number
                action = env.action_space.sample()                  # perform a random action
            else:
                action = np.argmax(q_table[current_state])          # perform the action from the current q-table
            
            observation, reward, done, info = env.step(action)      # perform the action on the environment
            next_state = Discrete(observation, bins)                # obtain the next state and discrete it for the q-table
            score += reward                                         # add the reward of this step to the total reward
            
            if not done:                                            # done = false: pole is still upright
                max_future_q = np.max(q_table[next_state])          # the action with the best q-value for the next state  
                current_q = q_table[current_state+(action,)]        # the q-value of the current state
                new_q = (1-lr)*current_q + lr*(reward + gamma*max_future_q) # q-learing formula
                q_table[current_state+(action,)] = new_q            # add the new q-value to the q-table

            current_state = next_state
            
        # done = TRUE: pole is not upright: episode stopped
        else:
            rewards += score                                        # the obtained reward
                   
            runs.append(score)                                      # store the score of this run in the array
            if score > 195 and steps >= 100 and solved == False:    # considered as solved: score is more than 195 (the first steps are skipped)
                solved = True
                print('Solved in episode : {} in time {}'.format(episode, (time.time()-ep_start)))
        
        # timestep value update
        if episode%timestep == 0:                                   # print every 1.000 steps
            print('Episode : {} | Reward -> {} | Max reward : {} | Time : {}'.format(episode,rewards/timestep, max(runs), time.time() - ep_start))
            data['max'].append(max(runs))                           # store at data: maximum score of all runs
            data['avg'].append(rewards/timestep)                    # store at data: average score
            if rewards/timestep >= 195: 
                print('Solved in episode : {}'.format(episode))
            rewards, runs= 0, [0] 
            
        # save the last 100 episodes for a plot
        if episode > episodes-100:
            lastepi.append(episodes-episode)
            lastscore.append(score);
        
        if episode > 200000:
            epsilon = 0.0;
               
    
    if len(ep) == len(data['max']):
        plt.plot(ep, data['max'], label = 'Max')
        plt.plot(ep, data['avg'], label = 'Avg')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend(loc = "upper left")
        
    env.close()
  
    
def main():         
    '''
    main function
    '''  
    
    # define the q-table
    q_table, bins = Qtable(len(env.observation_space.low), env.action_space.n)
    
    # start the training
    Q_learning(q_table, bins, lr = 0.15, gamma = 0.995, episodes = 300*10**3, timestep = 1000, epsilon = 0.20)
    
    # plot the results
    #epplot = [i for i in range(0, 300000 + 1, 1000)]
    #plt.plot(epplot, data['max'], label = 'Max')
    #plt.plot(epplot, data['avg'], label = 'Avg')
    #plt.xlabel('Episode')
    #plt.ylabel('Reward')
    #plt.legend(loc = "upper left")
    
    epplot = [i for i in range(0, 200000 + 1, 1000)]
    plt.plot(epplot, data['avg'][0:201], label = 'Avg')
    plt.show();
    
    epplot = [i for i in range(0, 100000 + 1, 1000)]
    plt.plot(epplot, data['avg'][200:301], label = 'Avg')
    plt.show();
    
    # plot the last 100 episodes
    pylab.plot(lastepi, lastscore, 'b');
    plt.show();
 
    
'''
main
'''
if __name__ == '__main__':
    main()  


