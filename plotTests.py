import matplotlib.pyplot as plt
import numpy as np
from QRunner import runQ
from MonteCarlo import runMC

iterations = 10
numEpisodes = 500
episodeLength = 1000
print("Start Q-Learning Sims...")
qWinPercents, qFirstWins, qWinsAfterFirst, qPostWinRates = runQ(iterations, numEpisodes, episodeLength, 50)
print()

print("Starting Monte Carlo Sims...")
mcWinPercents, mcFirstWins, mcWinsAfterFirst, mcPostWinRates = runMC(iterations, numEpisodes, episodeLength, 5)
print()

print("Q-Win Rates:")
print(qWinPercents)
print()

print("Q-Wins after first Win:")
print(qWinsAfterFirst)
print()

print("Q-Post Win Win Rates:")
print(qPostWinRates)
print()

print("MC-Win Rates:")
print(mcWinPercents)
print()

print("MC-Wins after first Win:")
print(mcWinsAfterFirst)
print()


print("MC-Post Win Win Rates:")
print(mcPostWinRates)
print()

# set width of bar 
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8)) 
  
# Set position of bar on X axis 
br1 = np.arange(iterations) 
br2 = [x + barWidth for x in br1] 
 
# Make the plot
plt.bar(br1, qWinPercents, color ='r', width = barWidth, 
        edgecolor ='grey', label ='Q-Learning') 
plt.bar(br2, mcWinPercents, color ='g', width = barWidth, 
        edgecolor ='grey', label ='Monte Carlo') 

for i, v in enumerate(qWinPercents):
    plt.text(br1[i] - 0.12, v + 0.02, int(qFirstWins[i]) if v != 0 else "NA")
    
for i, v in enumerate(mcWinPercents):
    plt.text(br2[i] - 0.12, v + 0.02, int(mcFirstWins[i]) if v != 0 else "NA") 

# Adding Xticks 
plt.xlabel('Iteration', fontweight ='bold', fontsize = 10) 
plt.ylabel('Win Rate', fontweight ='bold', fontsize = 10) 
plt.title("Win Rates for 10 Independent Iterations of 500 Episodes")
 
plt.legend()
plt.show()




    
  