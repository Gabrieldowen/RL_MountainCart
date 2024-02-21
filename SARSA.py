import numpy as np
import csv
import time
import pandas as pd
import matplotlib.pyplot as plt
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
		
		if env.render_mode == 'human':
			print(f"\n\n\n NEW EPISODE {episode} \n\n\n current epsilon: {epsilon}")
			print(f"\n totalHighScoreX: {HighScoreX}\n totalHighScoreVel: {HighScoreX} sizeofStates: {np.count_nonzero(stateTable)}")

			time.sleep(1)

		epsilon =  1 - (episode / numEpisodes)
		print(epsilon)
		stateMetrics, info = env.reset(seed=42)

		# get initial state and action
		state = (round(stateMetrics[0]*1000)+45) * abs(round(stateMetrics[1]*1000))
		action, policyChoice = policy(stateTable, state, epsilon)

		if env.render_mode == 'human':
			print(f"\n pos {round(stateMetrics[0]*100)+45} velo: {abs(round(stateMetrics[1]*2000))} result: {state}")

		episodeHighScoreX = 0
		episodeHighScoreVel = 0
		victory = False


		for _ in range(1000):


			
			# take action
			nextStateMetrics, reward, terminated, truncated, info = env.step(action)

			
			# gets the state as a result
			x = round(nextStateMetrics[0]*100)+45
			velocity = abs(round(nextStateMetrics[1]*1000))
			nextState = x * velocity

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

			# adjust the reward
			if x > 0:
				reward += (velocity*0.5) + (abs(x)*1.5)
			else:
				reward += (velocity*0.5)  + (abs(x))



			# get the next action
			nextAction, policyChoice = policy(stateTable, state, epsilon)

			if env.render_mode == 'human':
				print(f"\n ep: {episode+1} policy: {policyChoice} pos: {x} velo: {velocity} result: {state} print: {nextAction}")

			# q_table[state, action] += alpha * (reward + gamma * q_table[next_state, next_action] - q_table[state, action])

			stateTable[state, action] += alpha*(reward+gamma*stateTable[nextState, nextAction] - stateTable[state, action])
			
			state = nextState
			action = nextAction

			if terminated or truncated:
				print(f"\n\n GAME OVER FOR EPISODE {episode+1}\nterminated: {terminated}\ntruncated: {truncated}")

				with open('results/sarsaResults.csv', mode='a', newline='') as file:
				    writer = csv.writer(file)
				    writer.writerow([episode+1, terminated,episodeHighScoreX, episodeHighScoreVel, HighScoreX, HighScoreVel, epsilon, np.count_nonzero(stateTable)])
				break
				
	plotLearning(numEpisodes)
	return stateTable

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
	ax1.set_title('x learning rate')

	# Plot the second graph
	ax2.plot(episode, HighScoreVel, color='red')
	ax2.scatter(victoryRows['episode'], victoryRows['episodeHighScoreVel'], color='blue', label='Victory')
	ax2.set_xlabel('episode')
	ax2.set_ylabel('HighScoreVel')
	ax2.set_title('vel learning rate')

	# Adjust layout to prevent overlap
	plt.tight_layout()

	# Show the plots
	plt.show()


# epsilon greedy
def policy(stateTable, state, epsilon):
	if np.random.rand() > epsilon:
		return np.argmax(stateTable[state]), "follow"	
	else:
		return np.random.choice([0, 1, 2]), "explore"
		
