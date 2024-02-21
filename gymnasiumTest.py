import gymnasium as gym
import SARSA
env = gym.make("MountainCar-v0", render_mode='none')

env._max_episode_steps = 1000


SARSA.Learn(env, numEpisodes=100, epsilon=1)

env.close()