import numpy as np
import csv
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
# What each action does
	# 0: accelerate left
	# 1: Dont accelerate
	# 2: accelerate right

# nextStateMetrics [x-axis (-1.2 to 0.6), velocity(-0.07 to 0.07)]

def Learn(env, numEpisodes=100, initialEpsilon=1, alpha=0.1, gamma=0.99, stateTable=None):

	
	# create space for velocity and pos
	xSpace = np.linspace(-1.2, 0.6, 60)    # Between -1.2 and 0.6
	velSpace = np.linspace(-0.7, 0.07, 60)

	if stateTable is None:
		stateTable = np.zeros((len(xSpace),len(velSpace), 3)) 

	for episode in range(numEpisodes):

		# initialize total score
		totalReward = 0

		episodeHighScoreX = 0
		victory = False

		# update epsilon for next episode
		epsilon =  initialEpsilon / (episode+1)

		# extra items
		if env.render_mode == 'human':
			print(f"\n\n\n NEW EPISODE {episode} \n\n\n current epsilon: {epsilon}")
			time.sleep(1)

		# reset the game for the new episode
		stateMetrics, info = env.reset(seed=42)

		xBin = np.digitize(stateMetrics[0], xSpace)
		velBin = np.digitize(stateMetrics[1], velSpace)

		# get initial state and action
		action = policy(stateTable[xBin, velBin, :], epsilon)



		for _ in range(1000):


			
			# take action observe reward and nextState
			nextStateMetrics, reward, terminated, truncated, info = env.step(action)


			x = nextStateMetrics[0]
			velocity = nextStateMetrics[1]

			xBinNext = np.digitize(x, xSpace)
			velBinNext = np.digitize(velocity, velSpace)

			# Keeps episode highscore
			if x > episodeHighScoreX:
				episodeHighScoreX = x

			if x >=0.5:
				reward += 10	

			# keeps total reward per episode
			totalReward += reward	

			# get the next action from epsilon greedy policy
			nextAction = policy(stateTable[xBinNext, velBinNext, :], epsilon)

			# q_table[state, action] += alpha * (reward + gamma * q_table[next_state, next_action] - q_table[state, action])
			stateTable[xBin, velBin, action] += alpha*(reward+gamma*stateTable[xBinNext, velBin, nextAction] - stateTable[xBin, velBin, action])

			# moves to the next state			
			xBin = xBinNext
			velBin = velBinNext
			action = nextAction

			if terminated or truncated:
				print(f"\n\n GAME OVER FOR EPISODE {episode+1}\nterminated: {terminated}\ntruncated: {truncated}")

				if env.render_mode == 'none':
					with open('results/sarsaResults.csv', mode='a', newline='') as file:
					    writer = csv.writer(file)

					    if os.path.getsize('results/sarsaResults.csv') == 0:
	        				writer.writerow(['episode','victory','totalReward','episodeHighScoreX','epsilon\n'])

					    writer.writerow([episode+1, terminated,totalReward, episodeHighScoreX,epsilon])
				break
				
	return stateTable


# epsilon greedy
def policy(nextStateActions, epsilon):
	if np.random.rand() > epsilon:
		return np.argmax(nextStateActions)	
	else:
		return np.random.choice([0, 1, 2])


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


if __name__ == "__main__":
	plotLearning(1000)

		
