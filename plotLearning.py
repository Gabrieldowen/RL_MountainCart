import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plotLearning(numEpisodes, fileName):
	print("\nPlotting Learning...\n")
	df = pd.read_csv(f'results/{fileName}')
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

def plotLearningAverage(numEpisodes, fileName, numRuns):
	print("\nPlotting Learning...\n")

	# Read the entire CSV file
	df = pd.read_csv(f'results/{fileName}')

	indexArray = []
	for run in range(numRuns):
		indexArray.append(run*numEpisodes)

	meanArray = []
	# Select the rows and column, then calculate the mean
	for ep in range(numEpisodes):
		meanArray.append(df.iloc[indexArray, 2].mean())
		indexArray = [x + 1 for x in indexArray]


	# Plot the average reward for each episode
	plt.figure(figsize=(12, 6))
	plt.plot(range(1, numEpisodes + 1), meanArray, color='blue')

	plt.xlabel('Episode')
	plt.ylabel('Average Reward')
	plt.title('Average Reward for Each Episode Over 10 Runs')
	plt.show()

def plotTwoAverages(numEpisodes, fileName1, fileName2, numRuns):
	print("\nPlotting Learning...\n")

	# Read the entire CSV files
	df1 = pd.read_csv(f'results/{fileName1}')
	df2 = pd.read_csv(f'results/{fileName2}')

	indexArray = []
	for run in range(numRuns):
		indexArray.append(run*numEpisodes)

	meanArray1 = []
	meanArray2 = []
	# Calculate the mean for each episode for both files
	for ep in range(numEpisodes):
		meanArray1.append(df1.iloc[indexArray, 2].mean())
		meanArray2.append(df2.iloc[indexArray, 2].mean())
		indexArray = [x + 1 for x in indexArray]

	# Plot the average rewards for each episode
	plt.figure(figsize=(12, 6))
	plt.plot(range(1, numEpisodes + 1), meanArray1, color='blue', label='Eligibility Traces')
	plt.plot(range(1, numEpisodes + 1), meanArray2, color='red', label='No Eligibility Traces')

	plt.xlabel('Episode')
	plt.ylabel('Average Reward')
	plt.title('SARSA Average Reward for Each Episode Over 10 Runs')
	plt.legend()
	plt.show()

def plotWinRate():
	print(0)

if __name__ == "__main__":
	# plotTwoAverages(1000, "sarsaResults.csv","eligibityTraces.csv", 10)
	plotWinRate()