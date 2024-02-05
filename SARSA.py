import numpy as np

def Learn(env, numEpisodes=1, epsilon=0.1):

	table = np.zeros((2, 2)) #np.zeros((env.observation_space.n, env.action_space.n))

	for episode in range(numEpisodes):

		state, info = env.reset(seed=42)

		for _ in range(1000):

			# 0: accelerate left
			# 1: Dont accelerate
			# 2: accelerate right
			action = policy(table, state, epsilon, env) # env.action_space.sample()  # this is where you would insert your policy
			
			# state [x-axis, velocity]
			nextState, reward, terminated, truncated, info = env.step(action)
			print(f"a: {action}, ns: {state}, r: {reward}, i: {info}")


			state = nextState
			if terminated or truncated:
				break
				
	return table

# epsilon greedy
def policy(table, state, epsilon, env):
	if np.random.rand() > epsilon:
		print("\nfollow")
	else:
		print("\nexplore")

	action = env.action_space.sample()
	return action
