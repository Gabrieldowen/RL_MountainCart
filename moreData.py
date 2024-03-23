import matplotlib.pyplot as plt
import numpy as np

qPostWinRates = [0.8978102189781022, 0.9213759213759214, 0.9276807980049875, 0.8985507246376812, 0.9019607843137255, 0.9133663366336634, 0.9378109452736318, 0.914004914004914, 0.8913043478260869, 0.8948655256723717]
mcPostWinRates = [0.9504310344827587, 0.9423076923076923, 0.9911894273127754, 0.9556541019955654, 0.9507829977628636, 0.9482758620689655, 0.946236559139785, 0.8845315904139434, 0.9887640449438202, 0.9747126436781609]
iterations = len(qPostWinRates)


# set width of bar 
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8)) 
  
# Set position of bar on X axis 
br1 = np.arange(iterations) 
br2 = [x + barWidth for x in br1] 
 
# Make the plot
plt.bar(br1, qPostWinRates, color ='r', width = barWidth, 
        edgecolor ='grey', label ='Q-Learning') 
plt.bar(br2, mcPostWinRates, color ='g', width = barWidth, 
       edgecolor ='grey', label ='Monte Carlo') 

#plt.plot(qPostWinRates, label = "Q")
#plt.plot(mcPostWinRates, label = "MC")
#plt.ylim(0,1)
# Adding Xticks 
plt.xlabel('Iteration', fontweight ='bold', fontsize = 10) 
plt.ylabel('Win Rate', fontweight ='bold', fontsize = 10) 
plt.title("Win Rates after First Win for 10 Independent Iterations")
 
plt.legend()
plt.show()