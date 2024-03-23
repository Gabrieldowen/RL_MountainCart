import gymnasium as gym
import SARSA
import EligibilityTraces as ET
from plotLearning import plotLearning, plotLearningAverage 
import MonteCarlo
import QRunner

numEpisodes = 1000
numRuns = 1

choice = int(input("Enter 1 to run SARSA, \n2 to run with eligibility traces, \n3 to run Q-learning, \nor 4 to run Monte Carlo:\n "))

if choice == 1:
    for _ in range(numRuns):
        env = gym.make("MountainCar-v0", render_mode='none')
        env._max_episode_steps = 1000
        stateTable = SARSA.Learn(env, numEpisodes-1)
        env.close()

        # show agent without learning just following the stateTable
        env = gym.make("MountainCar-v0", render_mode='human')
        env._max_episode_steps = 1000
        SARSA.Learn(env, numEpisodes=1, initialEpsilon=0, alpha=0.1, gamma=0.99, stateTable = stateTable)
        env.close()
        plotLearningAverage(numEpisodes, 'sarsaResults.csv', numRuns)

elif choice == 2:
    for _ in range(numRuns):
        env = gym.make("MountainCar-v0", render_mode='none')
        env._max_episode_steps = 1000
        stateTable = ET.Learn(env, numEpisodes-1)
        env.close()

        # show agent without learning just following the stateTable
        env = gym.make("MountainCar-v0", render_mode='human')
        env._max_episode_steps = 1000
        ET.Learn(env, numEpisodes=1, initialEpsilon=0, alpha=0.1, gamma=0.99, stateTable = stateTable)
        env.close()
    plotLearningAverage(numEpisodes, 'eligibilityTraces.csv', numRuns)

elif choice == 3:
    QRunner.runQ(1, 1000, 500)

elif choice == 4:
    MonteCarlo.runMC(1, 1000, 5)

else:
    print("Invalid choice. Please enter 1 - 4.")


# plotLearningAverage( numEpisodes,'eligibilityTraces.csv', numRuns)