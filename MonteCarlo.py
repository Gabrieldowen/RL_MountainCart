import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from MonteCarloAgent import MonteCarloAgent

def runMC(iters, nEps, epLength, states = 5):
    env = gym.make('MountainCar-v0')

    epsilon = 1
    numEpisodes = nEps
    episodeLength = epLength
    numStates = states
    decayRate = 2 / numEpisodes 

    rewardsPerEpisode = []
    stepsPerEpisode = []
    iterations = iters

    winCounts = np.zeros(iterations)
    firstWins = np.zeros(iterations)
    winsAfterFirstWin = []

    for iter in range(iterations):
        print(f"Starting iteration {iter + 1}...")
        agent = MonteCarloAgent(env, epsilon, numEpisodes, episodeLength, numStates, decayRate)
        hasWon = False
        winsAfterFirstWin.append((0, 0))
        
        # For every episode
        for i in range(numEpisodes):
        
            # Get array of State, Action, and Reward for each episode
            stateActionRewards, stateActions, episodeReward, stepCount, won = agent.generateEpisode(i)
            
            if won:
                winCounts[iter] += 1
                if hasWon:
                    winsAfterFirstWin[iter] = (winsAfterFirstWin[iter][0] + 1, winsAfterFirstWin[iter][1])
                if not hasWon:
                    print(f"{i} Won!")
                    hasWon = True
                    firstWins[iter] = i + 1
                    winsAfterFirstWin[iter] = (winsAfterFirstWin[iter][0], int(numEpisodes - firstWins[iter]))  
            
            if i == numEpisodes / 4:
                print("   1/4 Done")
            elif i == numEpisodes / 2:
                print("   1/2 Done")
            elif i == 3 * (numEpisodes / 4):
                print("   3/4 Done")             

            # Update stats
            rewardsPerEpisode.append(episodeReward)
            stepsPerEpisode.append(stepCount)

            # For each pair s,a appearing in the episode    
            for step in range(len(stateActionRewards)):        
            
                # R <- return following the first occurence of s,a
                firstVisit = stateActions.index((stateActions[step]))
                R = stateActionRewards[len(stateActionRewards) - 1][4] - stateActionRewards[firstVisit][4]
            
                # Append R to Returns(s, a)
                agent.returns[stateActions[firstVisit]].append(R)
            
                # Q(s,a) <- average(Returns(s,a))
                agent.qTable[stateActions[firstVisit]] = np.mean(agent.returns[stateActions[firstVisit]])  
            
            # Decay epsilon
            agent.decayEpsilon()      

    #qtable = agent.qTable        

    #for i in range(len(winCounts)):
    #    winCounts[i] /= numEpisodes
        
    #postWinRates = []
    #for wins, runs in winsAfterFirstWin:
    #    if runs != 0: 
    #        postWinRates.append(wins / runs)    
    #    else:
    #        postWinRates.append(0)    
    
    print(sum(rewardsPerEpisode) / len(rewardsPerEpisode))
    
    """
    print("Win Rates:")
    print(winCounts)
    print()
    
    print("Wins after first Win:")
    print(winsAfterFirstWin)
    print()
    
    
    print("Post Win Win Rates:")
    print(postWinRates)
    print()
    """
        
    #return winCounts, firstWins, winsAfterFirstWin, postWinRates    

runMC(1, 2000, 200)


