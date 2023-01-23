import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
from core import mlp

def train(env_name='CartPole-v1', hidden_sizes=[32], lr=1e-2, 
          epochs=500, batch_size=5000, render=False):
    
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n
    
    policy_network = mlp([obs_dim]+hidden_sizes+[n_acts])
    optimizer = Adam(policy_network.parameters(), lr=lr)

    # Function for computing action distribution
    def get_policy_distribution(obs):
        logits = policy_network(obs)
        return Categorical(logits=logits)
    
    # Function for selecting action from the distribution 
    def get_actions(obs):
        distribution = get_policy_distribution(obs)
        return distribution.sample().item()

    # Function for computing loss, whose gradient is the policy gradient
    def compute_loss(obs, action, weights):
        logp = get_policy_distribution(obs).log_prob(action)
        return -(logp * weights).mean()

    def reward_to_go(ep_rewards):
        ep_len = len(ep_rewards)
        rtg = []
        for i in range(ep_len):
            rtg += [sum(ep_rewards[i:])]
        return rtg

    def train_one_epoch():
        batch_obs = []
        batch_actions = []
        batch_weights_rtg = []
        batch_rewards = []
        batch_ep_len = []
        
        # Reset episode variables
        obs = env.reset()
        done = False
        ep_rewards = []

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:
            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()
             
            # save obs
            batch_obs.append(obs)
            # act in the environment
            action = get_actions(torch.as_tensor(obs, dtype=torch.float32))
            obs, reward, done, _ = env.step(action)
            # save action, reward
            batch_actions.append(action)
            ep_rewards.append(reward)
            if done:
                # if episode is over, record info about episode
                episode_total_reward = sum(ep_rewards)
                epsiode_len = len(ep_rewards)
                batch_rewards.append(episode_total_reward)
                batch_ep_len.append(epsiode_len)
                # the weight for each logprob(a|s) is R(tau)
                batch_weights_rtg += reward_to_go(ep_rewards)
                # reset episode-specific variables
                obs = env.reset()
                ep_rewards = []
                done = False
                # won't render again this epoch  
                finished_rendering_this_epoch = True 
                # end experience loop if we have enough of it
                if len(batch_obs)>=batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(torch.as_tensor(batch_obs, dtype=torch.float32), 
                                  torch.as_tensor(batch_actions, dtype=torch.float32), 
                                  torch.as_tensor(batch_weights_rtg, dtype=torch.float32))
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rewards, batch_ep_len

    # Training loop
    for epoch in range(epochs):
        batch_loss, batch_rewards, batch_ep_len = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
            (epoch, batch_loss, np.mean(batch_rewards), np.mean(batch_ep_len)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing reward-to-go on simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)