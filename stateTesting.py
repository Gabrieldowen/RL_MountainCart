import matplotlib.pyplot as plt
import numpy as np
from QRunner import runQ
from MonteCarlo import runMC

iterations = 3
numEpisodes = 500
episodeLength = 1000
stateNums = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

qWinPercents = []
mcWinPercents = []

for states in stateNums:
    
    print("Start Q-Learning Sims...")
    qWinPercents2, qFirstWins, qWinsAfterFirst, qPostWinRates = runQ(iterations, numEpisodes, episodeLength, states)
    print()
    qWinPercents.append(sum(qWinPercents2) / len(qWinPercents2))

    print("Starting Monte Carlo Sims...")
    mcWinPercents2, mcFirstWins, mcWinsAfterFirst, mcPostWinRates = runMC(iterations, numEpisodes, episodeLength, states)
    print()
    mcWinPercents.append(sum(mcWinPercents2) / len(mcWinPercents2))
    
plt.plot(stateNums, qWinPercents, label = "Q-Learning")
plt.plot(stateNums, mcWinPercents, label = "Monte Carlo")    

plt.xlabel('State Division', fontweight ='bold', fontsize = 10) 
plt.ylabel('Win Rate', fontweight ='bold', fontsize = 10) 
plt.title("Win Rates per State Division")
 
plt.legend()
plt.show()