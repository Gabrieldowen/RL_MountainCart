import gymnasium as gym
import SARSA
import sandbox

env = gym.make("MountainCarContinuous-v0", render_mode="human")

#sandbox.sandbox(env)
SARSA.Learn(env)

env.close()