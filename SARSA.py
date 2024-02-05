import numpy as np

def Learn(env, numEpisodes=1):

	table = np.zeros((2, 2)) #np.zeros((env.observation_space.n, env.action_space.n))

	for episode in range(numEpisodes):

		observation, info = env.reset(seed=42)

		for _ in range(1000):

			# 0: accelerate left
			# 1: Dont accelerate
			# 2: accelerate right
			action = env.action_space.sample()  # this is where you would insert your policy
			
			# observation [x-axis, velocity]
			observation, reward, terminated, truncated, info = env.step(action)
			print(f"a: {action}, o: {observation}, r: {reward}, i: {info}")

			if terminated or truncated:
				break
				
	return table

