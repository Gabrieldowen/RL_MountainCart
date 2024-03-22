import gymnasium as gym
import SARSA
import EligibilityTraces as ET
from plotLearning import plotLearning, plotLearningAverage 


numEpisodes = 1000
numRuns = 10

for _ in range(numRuns):
    env = gym.make("MountainCar-v0", render_mode='none')
    env._max_episode_steps = 1000
    stateTable = ET.Learn(env, numEpisodes)
    env.close()
    """
    # train
    env = gym.make("MountainCar-v0", render_mode='none')
    env._max_episode_steps = 1000
    stateTable = SARSA.Learn(env, numEpisodes-1)
    env.close()

    # show agent without learning just following the stateTable
    env = gym.make("MountainCar-v0", render_mode='human')
    env._max_episode_steps = 1000
    SARSA.Learn(env, numEpisodes=1, initialEpsilon=0, alpha=0.1, gamma=0.99, stateTable = stateTable)
    env.close()
    """

plotLearningAverage( numEpisodes,'eligibilityTraces.csv', numRuns)