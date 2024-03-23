import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from QLearningAgent import QLearningAgent


def runQ(iters, nEps, epLength, states = 40):
    env = gym.make('MountainCar-v0')

    epsilon = 1
    numEpisodes = nEps
    learningRate = 0.9
    discount = 0.9
    numStates = states
    epsilonDecay = 2 / numEpisodes 
    iterations = iters

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
            
            if episode == numEpisodes / 4:
                print("   1/4 Done")
            elif episode == numEpisodes / 2:
                print("   1/2 Done")
            elif episode == 3 * (numEpisodes / 4):
                print("   3/4 Done") 
            
            
            obs, _ = env.reset()
            discreteObs = agent.discreteState(obs)
    
            terminated = False
        
            totalReward = 0
            stepCount = 0
        
            while not terminated and totalReward > -1 * epLength:
                # Get action and subsequent observation
                action = agent.getAction(discreteObs)
                nextObs, reward, terminated, _, _ = env.step(action)
                    
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
    
    print(sum(rewardsPerEpisode) / len(rewardsPerEpisode))
    
    #for i in range(len(winCounts)):
    #    winCounts[i] /= numEpisodes
    #postWinRates = []
    #for wins, runs in winsAfterFirstWin:
    #    if runs != 0:
    #        postWinRates.append(wins / runs)
    #    else:
    #        postWinRates.append(0)
    """
    print("Win Counts:")
    print(winCounts)
    print()
    
    print("Wins after first Win:")
    print(winsAfterFirstWin)
    print()
    
    
    print(postWinRates)        
    """
    #return winCounts, firstWins, winsAfterFirstWin, postWinRates   

runQ(1, 1000, 200)
"""
    monteWinPercentages = [0, 0, .49, .74, 0, 0, .37, 0, 0, 0]    
    monteFirstWins = [0, 0, 253, 126, 0,0, 200, 0, 0, 0]
# set width of bar 
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8)) 
 
# set height of bar 
 
# Set position of bar on X axis 
br1 = np.arange(len(winCounts)) 
br2 = [x + barWidth for x in br1] 
 
# Make the plot
plt.bar(br1, winCounts, color ='r', width = barWidth, 
        edgecolor ='grey', label ='Q-Learning') 
plt.bar(br2, monteWinPercentages, color ='g', width = barWidth, 
        edgecolor ='grey', label ='Monte Carlo') 

for i, v in enumerate(winCounts):
    plt.text(br1[i] - 0.12, v + 0.02, int(firstWins[i]) if v != 0 else "NA")
    
for i, v in enumerate(monteWinPercentages):
    plt.text(br2[i] - 0.12, v + 0.02, int(monteFirstWins[i]) if v != 0 else "NA") 

# Adding Xticks 
plt.xlabel('Iteration', fontweight ='bold', fontsize = 10) 
plt.ylabel('Win Rate', fontweight ='bold', fontsize = 10) 
plt.title("Win Rates for 10 Independent Iterations of 500 Episodes")
 
plt.legend()
plt.show()     
"""
