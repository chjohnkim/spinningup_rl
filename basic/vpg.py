import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
import core 

def train(env_name='CartPole-v1', hidden_sizes=[32], pi_lr=3e-2, vf_lr=1e-4, 
          epochs=500, train_v_iters=1, batch_size=5000, render=False,
          gamma=0.99, lam=0.95):
    
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n
    
    actor = core.mlp([obs_dim]+hidden_sizes+[n_acts])
    critic = core.mlp([obs_dim]+hidden_sizes+[1])

    pi_optimizer = Adam(actor.parameters(), lr=pi_lr)
    vf_optimizer = Adam(critic.parameters(), lr=vf_lr)

    # Function for computing action distribution
    def get_policy_distribution(obs):
        logits = actor(obs)
        return Categorical(logits=logits)
    
    # Function for selecting action from the distribution 
    def get_actions(obs):
        distribution = get_policy_distribution(obs)
        return distribution.sample().item()

    # Function for computing loss, whose gradient is the policy gradient
    def compute_loss_pi(obs, action, advantage):
        logp = get_policy_distribution(obs).log_prob(action)
        return -(logp * advantage).mean() 

    def compute_loss_vf(obs, rew_to_go):
        v_critic = critic(obs)
        return ((v_critic-rew_to_go)**2).mean()

    def train_one_epoch():
        batch_obs = []
        batch_actions = []
        batch_values = []
        batch_advantages = []
        batch_rewards = []
        batch_rtg = []
        batch_ep_len = []

        # Reset episode variables
        obs = env.reset()
        done = False
        ep_rewards = []
        ep_values = []
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
            # save action, value, reward
            batch_actions.append(action)
            ep_rewards.append(reward)
            value = critic(torch.as_tensor(obs, dtype=torch.float32))
            ep_values.append(value.item())
            if done:
                # if episode is over, record info about episode
                epsiode_len = len(ep_rewards)
                episode_total_reward = sum(ep_rewards)
                batch_rewards.append(episode_total_reward)
                batch_ep_len.append(epsiode_len)

                # Since episode is done, the final value and reward is 0
                ep_values.append(0)
                ep_rewards.append(0)
                
                ''' KEY PART: Computing Generalized Advantage Estimation'''
                # compute discounted rewards-to-go, to be targets for the value function
                batch_rtg += list(core.discount_cumsum(ep_rewards, gamma))[:-1]
                # the next two lines implement GAE-Lambda advantage calculation
                deltas = np.array(ep_rewards[:-1]) + gamma*np.array(ep_values[1:]) - np.array(ep_values[:-1])
                batch_advantages += list(core.discount_cumsum(deltas, gamma*lam))
                
                # reset episode-specific variables
                obs = env.reset()
                ep_rewards = []
                ep_values = []
                done = False
                # won't render again this epoch  
                finished_rendering_this_epoch = True 
                # end experience loop if we have enough of it
                if len(batch_obs)>=batch_size:
                    break        
        
        # Advantage normalization trick
        adv_mean, adv_std = np.mean(batch_advantages), np.std(batch_advantages)
        batch_adv_normalized = (np.asarray(batch_advantages) - adv_mean) / adv_std

        # take a single policy gradient update step
        pi_optimizer.zero_grad()
        batch_loss_pi = compute_loss_pi(torch.as_tensor(batch_obs, dtype=torch.float32), 
                                        torch.as_tensor(batch_actions, dtype=torch.float32), 
                                        torch.as_tensor(batch_adv_normalized, dtype=torch.float32))
        batch_loss_pi.backward()
        pi_optimizer.step()
        
        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            batch_loss_vf = compute_loss_vf(torch.as_tensor(batch_obs, dtype=torch.float32), 
                                            torch.as_tensor(batch_rtg, dtype=torch.float32))
            batch_loss_vf.backward()
            vf_optimizer.step()

        return batch_loss_pi, batch_rewards, batch_ep_len

    # Training loop
    for epoch in range(epochs):
        batch_loss, batch_rewards, batch_ep_len =train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
            (epoch, batch_loss, np.mean(batch_rewards), np.mean(batch_ep_len)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--pi_lr', type=float, default=3e-2)
    parser.add_argument('--vf_lr', type=float, default=1e-4)
    args = parser.parse_args()
    print('\nVanilla Policy Gradient.\n')
    train(env_name=args.env_name, render=args.render, pi_lr=args.pi_lr, vf_lr=args.vf_lr)