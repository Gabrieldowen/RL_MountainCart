import numpy as np
import csv
import time
import os
import gymnasium as gym
import EligibilityTraces as ET
from plotLearning import plotLearning 
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

		# update epsilon for next episode
		epsilon =  initialEpsilon / (episode+1)

		# reset the game for the new episode
		stateMetrics, info = env.reset(seed=42)

		xBin = np.digitize(stateMetrics[0], xSpace)
		velBin = np.digitize(stateMetrics[1], velSpace)

		# get initial state and action
		action = policy(stateTable[xBin, velBin, :], epsilon)



		for _ in range(1000):
			
			# take action observe reward and nextState
			nextStateMetrics, reward, terminated, truncated, info = env.step(action)

			# gets next state and places in correct bin
			x = nextStateMetrics[0]
			velocity = nextStateMetrics[1]

			xBinNext = np.digitize(x, xSpace)
			velBinNext = np.digitize(velocity, velSpace)

			# extra reward when you win
			if terminated:
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
				print(f"\n\n GAME OVER FOR EPISODE {episode+1}\nWin: {terminated}\nTime: {truncated}")

				with open('results/sarsaResults.csv', mode='a', newline='') as file:
					writer = csv.writer(file)

					if os.path.getsize('results/sarsaResults.csv') == 0:
						writer.writerow(['episode','victory','totalReward','epsilon\n'])

					writer.writerow([episode+1, terminated,totalReward,epsilon])
				break
				
	return stateTable


# epsilon greedy
def policy(nextStateActions, epsilon):
	if np.random.rand() > epsilon:
		return np.argmax(nextStateActions)	
	else:
		return np.random.choice([0, 1, 2])

		
if __name__ == "__main__":
	numEpisodes = 1000

	# train
	env = gym.make("MountainCar-v0", render_mode='none')
	env._max_episode_steps = 1000
	stateTable = Learn(env, numEpisodes)
	env.close()
