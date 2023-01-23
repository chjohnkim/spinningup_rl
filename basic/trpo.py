import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
import core 

def train(env_name='CartPole-v1', hidden_sizes=[32], vf_lr=1e-4, 
          epochs=500, train_v_iters=10, batch_size=5000, render=False,
          gamma=0.99, lam=0.95, delta=0.01):
    
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    actor = core.mlp([obs_dim]+hidden_sizes+[n_acts])
    critic = core.mlp([obs_dim]+hidden_sizes+[1])
    vf_optimizer = Adam(critic.parameters(), lr=vf_lr)

    # Function for computing action distribution
    def get_policy_distribution(obs):
        with torch.no_grad():
            logits = actor(obs)
        return Categorical(logits=logits)
    
    # Function for selecting action from the distribution 
    def get_actions(obs):
        distribution = get_policy_distribution(obs)
        return distribution.sample().item()

    # Function for computing action probability by applying softmax to logits
    def get_probabilities(obs):
        logits = actor(obs)
        return nn.functional.softmax(logits, dim=1)

    # Compute loss for value function
    def compute_loss_vf(obs, rew_to_go):
        v_critic = critic(obs)
        return ((v_critic-rew_to_go)**2).mean()

    def train_one_epoch():
        batch_obs = []
        batch_actions = []
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

        # Value function learning: Update critic
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            batch_loss_vf = compute_loss_vf(torch.as_tensor(batch_obs, dtype=torch.float32), 
                                            torch.as_tensor(batch_rtg, dtype=torch.float32))
            batch_loss_vf.backward()
            vf_optimizer.step()

        ''' KEY PART: Policy parameters are manually updated using dynamic step size '''
        # Compute probabilities of selected actions
        distribution = get_probabilities(torch.as_tensor(batch_obs, dtype=torch.float32))
        distribution = torch.distributions.utils.clamp_probs(distribution)
        probs = distribution[range(len(batch_obs)), batch_actions]

        # Compute the gradient wrt new probablities (surrogate function),
        # so second probablities should be treated as a constant
        L = surrogate_loss(probs, probs.detach(), torch.as_tensor(batch_adv_normalized, dtype=torch.float32))
        KL = kl_divergence(distribution, distribution)
        parameters = list(actor.parameters())
        g = flat_grad(L, parameters, retain_graph=True)
        # Create graph because we will call backward() on it for HVP
        d_kl = flat_grad(KL, parameters, create_graph=True) 
        
        def HVP(v):
            return flat_grad(d_kl@v, parameters, retain_graph=True)

        search_dir = conjugate_gradient(HVP, g)
        max_length = torch.sqrt(2*delta/(search_dir@HVP(search_dir)))
        max_step = max_length*search_dir

        def apply_update(grad_flattened):
            n = 0
            for p in actor.parameters():
                numel = p.numel()
                gf = grad_flattened[n:n+numel].view(p.shape)
                p.data+=gf
                n+=numel    
    
        def criterion(step):
            apply_update(step)
            with torch.no_grad():
                distribution_new = get_probabilities(torch.as_tensor(batch_obs, dtype=torch.float32))
                distribution_new = torch.distributions.utils.clamp_probs(distribution_new)
                probs_new = distribution_new[range(len(batch_obs)), batch_actions]
                L_new = surrogate_loss(probs_new, probs, torch.as_tensor(batch_adv_normalized, dtype=torch.float32))
                KL_new = kl_divergence(distribution, distribution_new)
            L_improvement = L_new - L
            if L_improvement>0 and KL_new<=delta:
                return True
            apply_update(-step)
            return False
        
        # Policy learning: Update actor 
        i = 0
        while not criterion((0.9**i)*max_step) and i<10:
            i+=1     
        return batch_rewards, batch_ep_len

    # Training loop
    for epoch in range(epochs):
        batch_rewards, batch_ep_len =train_one_epoch()
        print('epoch: %3d \t return: %.3f \t ep_len: %.3f'%
            (epoch, np.mean(batch_rewards), np.mean(batch_ep_len)))

''' KEY FUNCTIONS FOR TRPO'''
def surrogate_loss(new_probs, old_probs, advantages):
    return (new_probs / old_probs * advantages).mean()

def kl_divergence(p, q):
    p = p.detach()
    return (p * (p.log() - q.log())).sum(-1).mean()

def flat_grad(y, x, retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True
    g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
    g = torch.cat([t.view(-1) for t in g])
    return g

def conjugate_gradient(A, b, delta=0, max_iter=10):
    x, r, p = torch.zeros_like(b), b.clone(), b.clone()
    i = 0
    while i<max_iter:
        AVP = A(p)
        dot_old = r @ r
        alpha = dot_old / (p @ AVP)
        x_new = x + alpha*p
        if (x - x_new).norm() <= delta:
            return x_new
        
        i+=1
        r = r - alpha*AVP
        beta = (r @ r) / dot_old
        p = r + beta*p
        x=x_new
    return x    


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    print('\nTrust Region Policy Optimization.\n')
    train(env_name=args.env_name, render=args.render, 
          epochs=500, batch_size=5000, vf_lr=0.005, train_v_iters=1,
          gamma=0.99, lam=0.95, delta=0.01)


