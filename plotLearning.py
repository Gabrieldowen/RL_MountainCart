import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plotLearning(numEpisodes):
	df = pd.read_csv('results/sarsaResults.csv')
	lastSession = df.tail(numEpisodes)

	# Calculate the rolling average of total totalReward with a window size of 10
	rollingAvg = lastSession['totalReward'].rolling(window=10).mean()

	# Filter DataFrame to include only rows where 'victory' is true
	victoryRows = lastSession[lastSession['victory']]

	# Gets values to plot
	episode = lastSession['episode'].iloc[::10]  # Select every 10th episode
	totalReward = rollingAvg.iloc[::10]  # Select corresponding rolling average values

	# Create a figure and a single subplot
	fig, ax1 = plt.subplots(figsize=(12, 6))

	# Plot the data
	ax1.plot(episode, totalReward, color='blue')
	ax1.set_xlabel('Episode')
	ax1.set_ylabel('Total Reward')
	ax1.set_title('Average Total Reward Every 10 Episodes')

	firstVictoryIndex = victoryRows['episode'].iloc[0]

	# Add a vertical line at that index on the plot
	ax1.axvline(x=firstVictoryIndex, color='red', linestyle='--', label='First Victory')

	ax1.legend()

	# Show the plot
	plt.show()