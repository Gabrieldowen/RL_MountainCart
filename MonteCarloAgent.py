import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import itertools


class MonteCarloAgent():
    def __init__(self, env, numEpisodes, learningRate, discount, numStates, epsilon, decayRate):
        self.env = env
        self.numEpisodes = numEpisodes
        self.learningRate = learningRate
        self.discount = discount
        self.numStates = numStates
        self.epsilon = epsilon
        self.decayRate = decayRate

        # Create discrete state space. Velocity and Position are each divided into numStates    
        self.posSpace = np.linspace(env.observation_space.low[0], env.observation_space.high[0], numStates)    # Between -1.2 and 0.6
        self.velSpace = np.linspace(env.observation_space.low[1], env.observation_space.high[1], numStates)
        
        # Create the qTable (initialized to zero)
        self.qTable = np.zeros((len(self.posSpace), len(self.velSpace), env.action_space.n)) 
        
        # Create empty returns arrays for each state
        indices = list(itertools.product(range(len(self.posSpace)), range(len(self.velSpace)), range(env.action_space.n)))
        self.returns = {index: [] for index in indices}
        
        # Creating some stats arrays
        self.stepsPerEpisode = []
        self.rewardsPerEpisode = []
        
    # Gets discrete values from an observation (we use this to index on the qTable)    
    def discreteState(self, obs):
        return (np.digitize(obs[0], self.posSpace), np.digitize(obs[1], self.velSpace))
    
     # Gets an action based on Epsilon-greedy
    def getAction(self, discreteObs):
        # Explore
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        # Follow
        else:
            return np.argmax(self.qTable[discreteObs[0], discreteObs[1], :])
        
    def decayEpsilon(self):
        self.epsilon = max(self.epsilon - self.decayRate, 0)
    
    # Generates an episode and returns an array of the episode steps     
    def generateEpisode(self, episodeNum):
        terminated = False
        totalReward = 0
    
        obs, _ = self.env.reset()
        discreteObs = self.discreteState(obs)
        
        stateActionReward = []
        stateAction = []
        
        maxHeight = -1
        
        while not terminated and totalReward > -1000:
            
            # Get action and subsequent observation
            action = self.getAction(discreteObs)
            nextObs, reward, terminated, _, _ = self.env.step(action)
                        
            if nextObs[0] > maxHeight:
                maxHeight = nextObs[0]
            
            stateActionReward.append((discreteObs[0], discreteObs[1], action, reward))
            stateAction.append((discreteObs[0], discreteObs[1], action))
            
            if terminated:
                print("Episode #", episodeNum, "Won!")
        
            # Discretize next observation
            discreteNextObs = self.discreteState(nextObs)
            discreteObs = discreteNextObs
        
            # Update reward and 
            totalReward += reward
        
        if episodeNum % 100 == 0:
            print("Max Height", episodeNum, ":", maxHeight)
        return stateActionReward, stateAction
            