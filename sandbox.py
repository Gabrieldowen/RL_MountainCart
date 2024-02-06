def sandbox(env):

	observation, info = env.reset()

	for _ in range(1000):
	    action = env.action_space.sample()  # agent policy that uses the observation and info
	    print(f"\n{action} {type(action)}")
	    observation, reward, terminated, truncated, info = env.step(action)

	    if terminated or truncated:
	        observation, info = env.reset()

	env.close()