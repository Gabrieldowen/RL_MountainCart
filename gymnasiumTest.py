import gymnasium as gym
import SARSA
env = gym.make("MountainCar-v0", render_mode="human")
env._max_episode_steps = 700
SARSA.Learn(env)

env.close()