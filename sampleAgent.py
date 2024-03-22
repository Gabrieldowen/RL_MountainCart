import gymnasium as gym
import os
import csv
import plotLearning as PL

def Learn(numEpisodes):
    for episode in range(numEpisodes):

        observation, info = env.reset(seed=42)

        for _ in range(1000):
            action = env.action_space.sample()  # agent policy that uses the observation and info
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                print(f"\n\n GAME OVER FOR EPISODE {episode+1}\nWin: {terminated}\nTime: {truncated}")

                with open('results/sarsaResults.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)

                    if os.path.getsize('results/sampleAgentResults.csv') == 0:
                        writer.writerow(['episode','victory\n'])

                    writer.writerow([episode+1, terminated])
                break



if __name__ == "__main__":
    numEpisodes = 1000

    env = gym.make("MountainCar-v0", render_mode="None")
    env._max_episode_steps = 1000
    observation, info = env.reset()
    Learn(numEpisodes)

    env.close()