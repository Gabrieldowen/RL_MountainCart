import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from QLearningAgent import QLearningAgent

env = gym.make('MountainCar-v0')

epsilon = 1
numEpisodes = 1
learningRate = 0.9
discount = 0.9
numStates = 100
epsilonDecay = 150 / numEpisodes 

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
        
        print(reward)
        
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

#agent.plotRewards()
        
firstWin = [221, 216, 129, 121, 125, 101, 95, 103]
epsilonDec = [1/500, 2/500, 4/500, 8/500, 20/500, 40/500, 100/500, 150/500]      

plt.plot(epsilonDec, firstWin)
plt.suptitle("First Episode Won vs. Epsilon Decay per Episode")
plt.xlabel("Epsilon Decay per Episode") 
plt.ylabel("First Episode Won") 
plt.show()
