import gymnasium as gym
import SARSA
env = gym.make("MountainCar-v0", render_mode="human")

SARSA.Learn(env)

env.close()