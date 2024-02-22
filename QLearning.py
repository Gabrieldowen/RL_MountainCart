import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def run(episodes, is_training=True, render=False):
    env = gym.make('MountainCar-v0', render_mode='human' if render else None)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=episodes)

    # Divide position and velocity into segments
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)    # Between -1.2 and 0.6
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)    # Between -0.07 and 0.07
    
    q = np.zeros((len(pos_space), len(vel_space), env.action_space.n)) # init a 20x20x3 array

    learning_rate_a = 0.9 # alpha or learning rate
    discount_factor_g = 0.9 # gamma or discount factor.

    epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = 2 / episodes # epsilon decay rate
    rng = np.random.default_rng()   # random number generator

    episodeRewards = np.zeros(episodes)
    episodeStepCount = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]      # Starting position, starting velocity always 0
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)

        terminated = False          # True when reached goal

        totalReward = 0
        steps = 0
        
        while(not terminated and totalReward >- 1000):

            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_p, state_v, :])

            new_state, reward, terminated, _, _ = env.step(action)
            if terminated:
                print("Episode #", i, "won!")
            
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)

            q[state_p, state_v, action] = q[state_p, state_v, action] + learning_rate_a * (
                reward + discount_factor_g*np.max(q[new_state_p, new_state_v,:]) - q[state_p, state_v, action]
            )

            state = new_state
            state_p = new_state_p
            state_v = new_state_v

            totalReward += reward
            steps += 1

        epsilon = max(epsilon - epsilon_decay_rate, 0)
        episodeRewards[i] = totalReward
        episodeStepCount[i] = steps

    env.close()
    
    """"
    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])
        
    plt.plot(mean_rewards)
    plt.savefig(f'mountain_car.png')
    """
    smoothRewards = []
    for i in range(0, episodes, 10):
        avg = 0
        for j in range(i, i + 10):
            avg += episodeRewards[j]
        smoothRewards.append(avg / 10)
        
    smoothSteps = []
    for i in range(0, episodes, 10):
        avg = 0
        for j in range(i, i + 10):
            avg += episodeStepCount[j]
        smoothSteps.append(avg / 10)
        
    fig, axs = plt.subplots(ncols=2, figsize=(12, 5))    
        
    axs[0].plot(smoothRewards)
    axs[0].set_title("Rewards")
    #axs[0].show()
    
    axs[1].plot(smoothSteps)
    axs[1].set_title("Step Count")
    
    plt.show()

if __name__ == '__main__':
    run(500, is_training=True, render=False)

    #run(10, is_training=False, render=True)