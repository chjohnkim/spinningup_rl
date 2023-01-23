import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import time
MAX_ITERATIONS = 500
ENV = "FrozenLake-v1"

'''
ax = plt.subplot(1,1,1)
im = ax.imshow(np.zeros((10,10)))
plt.ion()
'''

def qlearning(env=ENV, epsilon=0.7, alpha=0.8, gamma=0.9):
    env = gym.make(env, is_slippery=False, render_mode="rgb_array")
    na = env.action_space.n
    ns = env.observation_space.n
    Q = np.zeros((ns, na))
    for i in range(MAX_ITERATIONS):
        # Write your code here to update Q
        # Reset ENV and get initial state 
        state, info = env.reset()
        # While episode not done do
        terminated = False
        step = 0
        while not terminated:
            # TODO: Sample action accoriding to epsilon-greedy policy
            if random.random() <= epsilon:
                action = random.choice(range(na))
            else:
                if np.sum(Q[state, :]) == 0:
                    action = random.choice(range(na))
                else:
                    action = np.argmax(Q[state, :], axis=0)
            # Get new state and reward
            state_prime, reward, terminated, truncated, info = env.step(action)
            print(f'Episode {i} step {step}, action {action}, reward {reward}')
            # Update Q[s,a] = (1-alpha)*Q[s,a] + alpha*(r + gamma*max(Q[s',a'])
            Q[state, action] = (1-alpha)*Q[state, action] + alpha*(reward + gamma*max(Q[state_prime, :]))
            # Update state
            state = state_prime
            step+=1
            #env_screen = env.render()
            #im.set_data(env_screen)
            #plt.pause(1e-6)
    return Q


if __name__ == '__main__':
    # Train Q with Q-learning
    Q = qlearning()
    # TODO: write your code here to evaluate and render environment
    env = gym.make(ENV, is_slippery=False, render_mode="rgb_array")
    state, info = env.reset()
    na = env.action_space.n
    terminated = False
    animation = []
    max_actions = 10
    counter = 0
    epsilon = 0.5
    while not terminated:
        # Select action accoriding to policy
        if random.random() <= epsilon:
            action = random.choice(range(na))
        else:
            if np.sum(Q[state, :]) == 0:
                action = random.choice(range(na))
            else:
                action = np.argmax(Q[state, :], axis=0)
        state, reward, terminated, truncated, info = env.step(action)
        env_screen = env.render()
        animation.append(env_screen)
        counter += 1
        if counter > max_actions:
            break
    f, axarr = plt.subplots(1, len(animation))

    for i in range(len(animation)):
        axarr[i].imshow(animation[i])
    plt.show()
