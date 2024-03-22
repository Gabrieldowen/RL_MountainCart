import gymnasium as gym
import SARSA
import EligibilityTraces as ET
from plotLearning import plotLearning 


numEpisodes = 1000
learningType = input("choose your learning type: \n 1: SARSA \n 2: SARSA with Eligibility Traces\n")

if learningType == "1":
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

    plotLearning(numEpisodes)

elif learningType == "2":
    # train
    env = gym.make("MountainCar-v0", render_mode='none')
    env._max_episode_steps = 1000
    stateTable = ET.Learn(env, numEpisodes-1)
    env.close()

    # show agent without learning just following the stateTable
    env = gym.make("MountainCar-v0", render_mode='human')
    env._max_episode_steps = 1000
    ET.Learn(env, numEpisodes=1, initialEpsilon=0, alpha=0.1, gamma=0.99, stateTable = stateTable)
    env.close()

    plotLearning(numEpisodes)