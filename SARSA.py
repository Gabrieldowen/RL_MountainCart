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

def Learn(env, numEpisodes=100, epsilon=1, alpha=0.1, gamma=0.99):

	

	stateTable = np.zeros((33000, 3)) 

	# initialize high score
	HighScoreX = 0
	HighScoreVel = 0

	for episode in range(numEpisodes):
		
		# extra items
		if env.render_mode == 'human':
			print(f"\n\n\n NEW EPISODE {episode} \n\n\n current epsilon: {epsilon}")
			print(f"\n totalHighScoreX: {HighScoreX}\n totalHighScoreVel: {HighScoreX} sizeofStates: {np.count_nonzero(stateTable)}")

			time.sleep(1)
		episodeHighScoreX = 0
		episodeHighScoreVel = 0
		victory = False

		# update epsilon for next episode
		epsilon =  1 - (episode / numEpisodes)

		# reset the game for the new episode
		stateMetrics, info = env.reset(seed=42)

		# get initial state and action
		state = int(str(round(stateMetrics[0]*100)+45) + str(abs(round(stateMetrics[1]*200))))
		action, policyChoice = policy(stateTable, state, epsilon)



		for _ in range(1000):


			
			# take action observe reward and nextState
			nextStateMetrics, reward, terminated, truncated, info = env.step(action)

			# adjust/scale the next state values
			x = round(nextStateMetrics[0]*100)+45
			velocity = abs(round(nextStateMetrics[1]*200))
			nextState = int(str(x) + str(velocity))

			# adjust the reward
			if x > 0:
				reward += nextStateMetrics[1] + nextStateMetrics[0]
			else:
				reward += nextStateMetrics[0] + abs(nextStateMetrics[0])

			# Keeps total highscores
			if x > HighScoreX:
				HighScoreX = x
			if velocity > HighScoreVel:
				HighScoreVel = velocity

			# Keeps episode highscore
			if x > episodeHighScoreX:
				episodeHighScoreX = x
			if velocity > episodeHighScoreVel:
				episodeHighScoreVel = velocity


			# get the next action from epsilon greedy policy
			nextAction, policyChoice = policy(stateTable, state, epsilon)

			if env.render_mode == 'human':
				print(f"\n ep: {episode+1} policy: {policyChoice} pos: {x} velo: {velocity} state: {state} n/a: {nextAction}")

			# q_table[state, action] += alpha * (reward + gamma * q_table[next_state, next_action] - q_table[state, action])

			stateTable[state, action] += alpha*(reward+gamma*stateTable[nextState, nextAction] - stateTable[state, action])

			# moves to the next state			
			state = nextState
			action = nextAction

			if terminated or truncated:
				print(f"\n\n GAME OVER FOR EPISODE {episode+1}\nterminated: {terminated}\ntruncated: {truncated}")

				with open('results/sarsaResults.csv', mode='a', newline='') as file:
				    writer = csv.writer(file)

				    if os.path.getsize('results/sarsaResults.csv') == 0:
        				writer.writerow(['episode','victory','episodeHighScoreX','episodeHighScoreVel','HighScoreX','HighScoreVel','epsilon','stateTableSize1\n'])

				    writer.writerow([episode+1, terminated,episodeHighScoreX, episodeHighScoreVel, HighScoreX, HighScoreVel, epsilon, np.count_nonzero(stateTable)])
				break
				
	plotLearning(numEpisodes)
	return stateTable

# epsilon greedy
def policy(stateTable, state, epsilon):
	if np.random.rand() > epsilon:
		return np.argmax(stateTable[state]), "follow"	
	else:
		return np.random.choice([0, 1, 2]), "explore"

def plotLearning(numEpisodes):
	df = pd.read_csv('results/sarsaResults.csv')
	lastSession = df.tail(numEpisodes)

	# Filter DataFrame to include only rows where 'victory' is true
	victoryRows = lastSession[lastSession['victory']]

	# gets values to plot
	episode = lastSession['episode']
	HighScoreX = lastSession['episodeHighScoreX']
	HighScoreVel = lastSession['episodeHighScoreVel']


	# Create a figure and two subplots
	fig, (ax1, ax2) = plt.subplots(2)

	# Plot the first graph
	ax1.plot(episode, HighScoreX, color='blue')
	ax1.scatter(victoryRows['episode'], victoryRows['episodeHighScoreX'], color='red', label='Victory')
	ax1.set_xlabel('episode')
	ax1.set_ylabel('HighScoreX')
	ax1.set_title('highest x per episode')

	# Plot the second graph
	ax2.plot(episode, HighScoreVel, color='red')
	ax2.scatter(victoryRows['episode'], victoryRows['episodeHighScoreVel'], color='blue', label='Victory')
	ax2.set_xlabel('episode')
	ax2.set_ylabel('HighScoreVel')
	ax2.set_title('highest velocity per episode')

	# Adjust layout to prevent overlap
	plt.tight_layout()

	# Show the plots
	plt.show()


		
