import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def run(episodes, is_training=True, render=False):
    env = gym.make('MountainCar-v0', render_mode='human' if render else None)
    env._max_epsiode_steps = 1000
    #env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=episodes)

    # Divide position and velocity into segments
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)    # Between -1.2 and 0.6
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)    # Between -0.07 and 0.07
    
    # init a 20x20x3 array
    q = np.zeros((len(pos_space), len(vel_space), env.action_space.n)) 

    # alpha or learning rate
    learningRate = 0.9 
    
    # gamma or discount factor.
    discount = 0.9 

    # 1 = 100% random action at the beginning
    epsilon = 1        
    epsilonDecay = 2 / episodes 
    rng = np.random.default_rng() 

    episodeRewards = np.zeros(episodes)
    episodeStepCount = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]      # Starting position, starting velocity always 0
        pDiscrete = np.digitize(state[0], pos_space)
        vDiscrete = np.digitize(state[1], vel_space)

        terminated = False # True when reached goal

        totalReward = 0
        steps = 0
        
        while not terminated and totalReward > -1000:

            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[pDiscrete, vDiscrete, :])

            new_state, reward, terminated, _, _ = env.step(action)
            if terminated:
                print("Episode #", i, "won!")
            
            new_pDiscrete = np.digitize(new_state[0], pos_space)
            new_vDiscrete = np.digitize(new_state[1], vel_space)

            q[pDiscrete, vDiscrete, action] = q[pDiscrete, vDiscrete, action] + learningRate * (
                reward + discount*np.max(q[new_pDiscrete, new_vDiscrete,:]) - q[pDiscrete, vDiscrete, action]
            )

            pDiscrete = new_pDiscrete
            vDiscrete = new_vDiscrete

            totalReward += reward
            steps += 1
            
        epsilon = max(epsilon - epsilonDecay, 0)
        episodeRewards[i] = totalReward
        episodeStepCount[i] = steps

    env.close()
    
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
        
    _, axs = plt.subplots(ncols=2, figsize=(12, 5))    
        
    axs[0].plot(smoothRewards)
    axs[0].set_title("Rewards")
    
    axs[1].plot(smoothSteps)
    axs[1].set_title("Step Count")
    
    plt.show()

if __name__ == '__main__':
    run(500, is_training=True, render=False)

    #run(10, is_training=False, render=True)