import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from MonteCarloAgent import MonteCarloAgent

env = gym.make('MountainCar-v0')

epsilon = 1
numEpisodes = 3000
learningRate = 0.9
discount = 0.9
numStates = 30
decayRate = 2 / numEpisodes 

agent = MonteCarloAgent(env, numEpisodes, learningRate, discount, numStates, epsilon, decayRate)

for i in range(numEpisodes):
    stateActionRewards, stateActions = agent.generateEpisode(i)
    # goal = 0
    
    for step in range(len(stateActionRewards)):        
        
        # goal = discount * goal + stateActionRewards[step]
        # R <- return following the first occurence of s,a
        firstVisit = stateActions.index((stateActions[step]))
        R = 0
        for step in range(firstVisit + 1, len(stateActionRewards)):
            R += stateActionRewards[step][3]
        
        # Append R to Returns(s, a)
        agent.returns[stateActions[step]].append(R)
        
        # Q(s,a) <- average(Returns(s,a))
        agent.qTable[stateActions[step]] = sum(agent.returns[stateActions[step]]) / len(agent.returns[stateActions[step]])     
        
        # Decay epsilon
        agent.decayEpsilon()   
