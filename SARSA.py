import numpy as np
import time
# What each action does
	# 0: accelerate left
	# 1: Dont accelerate
	# 2: accelerate right

# nextStateMetrics [x-axis (-1.2 to 0.6), velocity(-0.07 to 0.07)]

def Learn(env, numEpisodes=3, epsilon=0.50, alpha=0.1, gamma=0.99):

	

	stateTable = np.zeros((8250, 3)) 

	# initialize high score
	HighScoreX = 0
	HighScoreVel = 0

	for episode in range(numEpisodes):
		
		print(f"\n\n\n NEW EPISODE \n\n\n current epsilon: {epsilon}")
		time.sleep(1)

		
		print(epsilon)
		stateMetrics, info = env.reset(seed=42)

		# get initial state and action
		state = (round(stateMetrics[0]*100)+45) * abs(round(stateMetrics[1]*1000))
		action = np.array([policy(stateTable, state, epsilon)-1])
		print(f"\n pos {round(stateMetrics[0]*100)+45} velo: {abs(round(stateMetrics[1]*1000))} result: {state}")


		for _ in range(1000):


			
			# take action
			nextStateMetrics, reward, terminated, truncated, info = env.step(action)

			print(f"\n\n{nextStateMetrics[0]} {type(nextStateMetrics[0])}\n\n")
			
			# gets the state as a result
			x = int(np.round(nextStateMetrics[0]*100)+45)
			velocity = int(abs(np.round(nextStateMetrics[1]*1000)))
			nextState = x * velocity

			# Keeps highscores
			if x > HighScoreX:
				HighScoreX = x
			if velocity > HighScoreVel:
				HighScoreVel = velocity

			# adjust the reward
			if x > 0:
				reward += velocity  * 10+ (abs(x) )**2  
			else:
				reward += velocity * 10 + (abs(x) )  


			# get the next action
			nextAction = policy(stateTable, state, epsilon)

			print(f"\n pos {x} velo: {velocity} result: {state}")

			# q_table[state, action] += alpha * (reward + gamma * q_table[next_state, next_action] - q_table[state, action])

			stateTable[state, action[0]] += alpha*(reward+gamma*stateTable[nextState, nextAction[0]] - stateTable[state, action[0]])
			
			state = nextState
			action = nextAction

			if terminated or truncated:
				epsilon -= 0.25
				print(f"\n\n GAME OVER FOR \nterminated: {terminated}\ntruncated: {truncated}")
				print(f"\n HighScoreX: {HighScoreX}\n HighScoreVel: {HighScoreX} sizeofStates: {np.count_nonzero(stateTable)}")
				break
				
	return stateTable

# epsilon greedy
def policy(stateTable, state, epsilon):
	if np.random.rand() > epsilon:
		print(f"\nfollow ")
		return np.array([np.random.choice(len(stateTable[state] -1))])
	else:
		print(f"\nexplore")
		return np.array([np.argmax(stateTable[state])-1])	
