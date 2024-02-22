import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from QLearningAgent import QLearningAgent

env = gym.make('MountainCar-v0')

epsilon = 1
numEpisodes = 300
learningRate = 0.9
discount = 0.9
numStates = 20
epsilonDecay = 2 / numEpisodes 

agent = QLearningAgent(env, epsilon, numEpisodes, learningRate, discount, numStates, epsilonDecay)

for episode in range(numEpisodes):
    obs, _ = env.reset()
    discreteObs = agent.discreteState(obs)
  
    terminated = False
    
    totalReward = 0
    stepCount = 0
    
    while not terminated and totalReward > -1000:
        # Get action and subsequent observation
        action = agent.getAction(discreteObs)
        nextObs, reward, terminated, _, _ = env.step(action)
        if terminated:
            print("Episode #", episode, "Won!")
        
        # Discretize next observation and update qTable
        discreteNextObs = agent.discreteState(nextObs)
        agent.updateQTable(reward, discreteObs, discreteNextObs, action)
        
        discreteObs = discreteNextObs
        
        # Update reward and stepCount
        totalReward += reward
        stepCount += 1
        
    agent.decayEpsilon()
    agent.updateTotals(episode, totalReward, stepCount)

agent.plot()
        
        
