import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from QLearningAgent import QLearningAgent


def runQ(iters, nEps, epLength, states = 40):
    env = gym.make('MountainCar-v0')

    # Agent parameters
    epsilon = 1
    numEpisodes = nEps
    learningRate = 0.9
    discount = 0.9
    numStates = states
    epsilonDecay = 2 / numEpisodes 
    iterations = iters

    # Some stat stuff
    rewardsPerEpisode = []
    firstWins = np.zeros(iterations)
    winCounts = np.zeros(iterations)
    winsAfterFirstWin = []

    for iter in range(iterations):
        print(f"Starting iteration {iter}...")
        agent = QLearningAgent(env, epsilon, numEpisodes, learningRate, discount, numStates, epsilonDecay)
        hasWon = False
        winsAfterFirstWin.append((0,0))
        
        for episode in range(numEpisodes):
            
            # Print out progress for a given iteration
            if episode == numEpisodes / 4:
                print("   1/4 Done")
            elif episode == numEpisodes / 2:
                print("   1/2 Done")
            elif episode == 3 * (numEpisodes / 4):
                print("   3/4 Done") 
            
            # Ready environment and other variables
            obs, _ = env.reset()
            discreteObs = agent.discreteState(obs)
            terminated = False
            totalReward = 0
            stepCount = 0
        
            # Run an episode
            while not terminated and totalReward > -1 * epLength:
                # Get action and subsequent observation
                action = agent.getAction(discreteObs)
                nextObs, reward, terminated, _, _ = env.step(action)
                    
                # Used for experiments    
                if terminated:
                    winCounts[iter] += 1
                    if hasWon:
                        winsAfterFirstWin[iter] = (winsAfterFirstWin[iter][0] + 1, winsAfterFirstWin[iter][1])
                    if not hasWon:
                        print(f"{episode} Won!")
                        hasWon = True
                        firstWins[iter] = episode + 1
                        winsAfterFirstWin[iter] = (winsAfterFirstWin[iter][0], int(numEpisodes - firstWins[iter]))    
                
                # Discretize next observation and update qTable
                discreteNextObs = agent.discreteState(nextObs)
                agent.updateQTable(reward, discreteObs, discreteNextObs, action)
            
                discreteObs = discreteNextObs
            
                # Update reward and stepCount
                totalReward += reward
                stepCount += 1
            
            rewardsPerEpisode.append(totalReward)
            agent.decayEpsilon()
            agent.updateTotals(episode, totalReward, stepCount)
    
if __name__ == '__main__':
    # Run one time, with an 1000 episodes 200 steps in length
    runQ(1, 1000, 200)
