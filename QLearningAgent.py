import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

class QLearningAgent():
    def __init__(self, env, epsilon, numEpisodes, learningRate, discount, numStates, decayRate):
        self.env = env
        self.epsilon = epsilon
        self.numEpisodes = numEpisodes
        self.learningRate = learningRate
        self.discount = discount
        self.numStates = numStates
        self.decayRate = decayRate

        # Create discrete state space. Velocity and Position are each divided into numStates    
        self.posSpace = np.linspace(env.observation_space.low[0], env.observation_space.high[0], numStates)
        self.velSpace = np.linspace(env.observation_space.low[1], env.observation_space.high[1], numStates)
        
        # Create the qTable (initialized to zero)
        self.qTable = np.zeros((len(self.posSpace), len(self.velSpace), env.action_space.n)) 
        
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
        
    def updateQTable(self, reward, discreteObs: tuple[int, int], nextDiscreteObs: tuple[int, int], action):
        
        # Calculates temporal distance
        TD = np.max(self.qTable[nextDiscreteObs[0], nextDiscreteObs[1], :]) - self.qTable[discreteObs[0], discreteObs[1], action]
        
        # Updates qTable for current observation
        self.qTable[discreteObs[0], discreteObs[1], action] += self.learningRate * reward + self.discount * TD
        
    # Updates episodic stats    
    def updateTotals(self, episode, totalReward, stepCount):
        self.rewardsPerEpisode.append(totalReward)
        self.stepsPerEpisode.append(stepCount)  
    
    # Slowly shifts actions from explore to exploit    
    def decayEpsilon(self):
        self.epsilon = max(self.epsilon - self.decayRate, 0)
     