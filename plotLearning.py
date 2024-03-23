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
	plt.title(f'Average Reward for Each Episode Over {numRuns} Runs')
	plt.show()

def plotAverages(numEpisodes, fileName1, fileName2, fileName3, fileName4, fileName5, numRuns):
	print("\nPlotting Learning...\n")

	# Read the entire CSV files
	df1 = pd.read_csv(f'results/{fileName1}')
	df2 = pd.read_csv(f'results/{fileName2}')
	df3 = pd.read_csv(f'results/{fileName3}')
	df4 = pd.read_csv(f'results/{fileName4}')
	df5 = pd.read_csv(f'results/{fileName5}')

	indexArray = []
	for run in range(numRuns):
		indexArray.append(run*numEpisodes)

	meanArray1 = []
	meanArray2 = []
	meanArray3 = []
	meanArray4 = []
	meanArray5 = []
	# Calculate the mean for each episode for all five files
	for ep in range(numEpisodes):
		meanArray1.append(df1.iloc[indexArray, 2].mean())
		meanArray2.append(df2.iloc[indexArray, 2].mean())
		meanArray3.append(df3.iloc[indexArray, 2].mean())
		meanArray4.append(df4.iloc[indexArray, 2].mean())
		meanArray5.append(df5.iloc[indexArray, 2].mean())
		indexArray = [x + 1 for x in indexArray]

	# Calculate the average window for each episode
	avg_window1 = [np.mean(meanArray1[max(0, i - 10):i+1]) for i in range(len(meanArray1))]
	avg_window2 = [np.mean(meanArray2[max(0, i - 10):i+1]) for i in range(len(meanArray2))]
	avg_window3 = [np.mean(meanArray3[max(0, i - 10):i+1]) for i in range(len(meanArray3))]
	avg_window4 = [np.mean(meanArray4[max(0, i - 10):i+1]) for i in range(len(meanArray4))]
	avg_window5 = [np.mean(meanArray5[max(0, i - 10):i+1]) for i in range(len(meanArray5))]

	# Plot the average rewards for each episode
	plt.figure(figsize=(12, 6))
	plt.plot(range(1, numEpisodes + 1), avg_window1, color='blue', label='gamma = 0.99')
	plt.plot(range(1, numEpisodes + 1), avg_window2, color='red', label='gamma = 0.5')
	plt.plot(range(1, numEpisodes + 1), avg_window3, color='green', label='gamma = 0.45')
	plt.plot(range(1, numEpisodes + 1), avg_window4, color='orange', label='gamma = 0.44')
	plt.plot(range(1, numEpisodes + 1), avg_window5, color='purple', label='gamma = 0.42')

	plt.xlabel('Episode')
	plt.ylabel('Average Reward')
	plt.title('SARSA Average Reward for Each Episode Over 10 Runs')
	plt.legend()
	plt.show()

if __name__ == "__main__":
	# plotTwoAverages(1000, "sarsaResults.csv","eligibityTraces.csv", 10)
	plotAverages(1000, "sarsaResults.csv","sarsaResults_g5.csv","sarsaResults_g45.csv", "sarsaResults_g44.csv",  "sarsaResults_g42.csv",  10)