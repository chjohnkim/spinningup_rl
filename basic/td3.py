import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Box
from copy import deepcopy
import core 

class ActorCritic:
    def __init__(self, obs_dim, act_dim, hidden_sizes=[256, 256]):
        self.pi = core.mlp([obs_dim]+hidden_sizes+[act_dim], output_activation=nn.Tanh)
        self.q1 = core.mlp([obs_dim+act_dim]+hidden_sizes+[1])
        self.q2 = core.mlp([obs_dim+act_dim]+hidden_sizes+[1])
        
    def act(self, obs):
        with torch.no_grad():
            action = self.pi(obs)
        return action.numpy()

    def get_q1(self, obs, action):
        q = self.q1(torch.cat([obs, action], dim=-1)) 
        return torch.squeeze(q, -1) 

    def get_q2(self, obs, action):
        q = self.q2(torch.cat([obs, action], dim=-1)) 
        return torch.squeeze(q, -1) 


def train(env_name='HalfCheetah-v2', hidden_sizes=[256, 256], replay_size=int(1e6), 
          epochs=500, batch_size=100, steps_per_epoch=4_000, start_steps=10_000,
          update_after=1_000, update_every=50, update_iters=50,   
          pi_lr=1e-3, q_lr=1e-3, gamma=0.99, polyak=0.995, 
          act_noise=0.1, target_noise=0.2, noise_clip=0.5, policy_delay=2,
          render_every=1, render=False):

    env = gym.make(env_name)
    test_env = gym.make(env_name)

    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Box), \
        "This example only works for envs with continuous action spaces."

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_ep_len = env._max_episode_steps
    # Action limit for clamping, critically, assumes all dimensions share the same bound
    act_limit = env.action_space.high[0]

    # Create actor critic model and target networks
    ac = ActorCritic(obs_dim, act_dim, hidden_sizes)
    ac_target = deepcopy(ac)

    # Freeze target networks with respect to optimizers (because we only update via polyak averaging)
    for p in ac_target.pi.parameters():
        p.requires_grad = False
    for p in ac_target.q1.parameters():
        p.requires_grad = False
    for p in ac_target.q2.parameters():
        p.requires_grad = False
    
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_params = list(ac.q1.parameters()) + list(ac.q2.parameters())
    q_optimizer = Adam(q_params, lr=q_lr) # Need to pass both q params somehow


    # Instantiate replay buffer object
    replay_buffer = core.ReplayBuffer(obs_dim, act_dim, replay_size)

    # Used for uniformly random sampling action at early training stage
    def get_policy_action(obs, noise_scale):
        action = ac.act(torch.as_tensor(obs, dtype=torch.float32))
        action += noise_scale * np.random.randn(act_dim)
        return np.clip(action, -act_limit, act_limit)

    ''' KEY PART: Handles twin Q functions '''
    # Function for computing TD3 Q-loss
    # Takes smaller q value between q1 and q2
    def compute_loss_q(data):
        obs = data['obs']
        action = data['act']
        reward = data['rew']
        next_obs = data['next_obs']
        done = data['done']
        ''' Twin Q'''
        q1 = ac.get_q1(obs, action) 
        q2 = ac.get_q2(obs, action)

        # Bellman backup for Q function
        with torch.no_grad():
            pi_target = ac_target.pi(next_obs)

            ''' Target policy smoothing '''
            epsilon = torch.randn_like(pi_target) * target_noise
            epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
            next_target_action = pi_target + epsilon
            next_target_action = torch.clamp(next_target_action, -act_limit, act_limit)

            ''' Target Q-values '''            
            q1_pi_target = ac_target.get_q1(next_obs, next_target_action)
            q2_pi_target = ac_target.get_q2(next_obs, next_target_action)
            q_pi_target = torch.min(q1_pi_target, q2_pi_target)
            backup = reward + gamma * (1-done) * q_pi_target

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2
        return loss_q

    # Function for computing DDPG pi loss (no change from DDPG)
    def compute_loss_pi(data):
        obs = data['obs']
        q1_pi = ac.get_q1(obs, ac.pi(obs))
        return -q1_pi.mean()


    ''' 
    The difference from DDPG is that two Q functions needs to be updated.
    The policy network and target networks are updated less frequently.
    '''
    def update(data, timer):
        # First run one gradient step for Q
        q_optimizer.zero_grad()
        loss_q = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        ''' Updates policy and target networks less frequently '''
        if timer%policy_delay==0:
            # Freeze Q-network so you don't waste comp effort 
            # computing gradients for it during policy learning step
            for p in ac.q1.parameters():
                p.requires_grad = False
            for p in ac.q2.parameters():
                p.requires_grad = False
            
            # Run one gradient descent step for pi
            pi_optimizer.zero_grad()
            loss_pi = compute_loss_pi(data)
            loss_pi.backward()
            pi_optimizer.step()

            # Unfreeze Q-network so you can optimize at next update step
            for p in ac.q1.parameters():
                p.requires_grad = True
            for p in ac.q2.parameters():
                p.requires_grad = True

            # Finally, update target networks by polyak averaging
            with torch.no_grad():
                for p, p_target in zip(ac.q1.parameters(), ac_target.q1.parameters()):
                    p_target.data.mul_(polyak)
                    p_target.data.add_((1-polyak)*p.data)
                for p, p_target in zip(ac.q2.parameters(), ac_target.q2.parameters()):
                    p_target.data.mul_(polyak)
                    p_target.data.add_((1-polyak)*p.data)
                for p, p_target in zip(ac.pi.parameters(), ac_target.pi.parameters()):
                    p_target.data.mul_(polyak)
                    p_target.data.add_((1-polyak)*p.data)
            
    # Logging episode info, resets every new epoch
    log_ep_len = []
    log_ep_rew = []

    # Reset episode variables
    obs = env.reset()
    done = False
    ep_reward = 0
    ep_len = 0
    # Main loop: Collect experience in environment and update policy
    total_steps = steps_per_epoch * epochs
    for step in range(total_steps):
        # Until start steps have elapsed, randomly sample actions 
        # from uniform distribution for better exploration. 
        # Afterwards used learned policy with noise
        if step>start_steps:
            action = get_policy_action(obs, act_noise)
        else:
            action = env.action_space.sample()

        # Step the environment
        next_obs, reward, done, _ = env.step(action)
        ep_len+=1
        ep_reward+=reward
        # Ignore done signal if it comes from max time horizon of episode
        done = False if ep_len==max_ep_len else done # TODO: Check if timed correctly
        # Store trajectory into replay buffer            
        replay_buffer.store(obs, action, reward, next_obs, done)
        obs = next_obs

        # Handle end of episode or reached trajectory max len 
        if done or ep_len==max_ep_len:
            log_ep_len.append(ep_len)
            log_ep_rew.append(ep_reward)
            # reset episode-specific variables
            obs = env.reset()
            done = False
            ep_len = 0
            ep_reward = 0            
        
        # Update every interval after enough steps have been taken
        if step>=update_after and step%update_every==0:
            for iter in range(update_iters): 
                batch_dict = replay_buffer.sample_batch(batch_size)
                update(batch_dict, iter)
      
        # Epoch handling
        if (step+1)%steps_per_epoch==0:
            epoch = (step+1) // steps_per_epoch 
            print('epoch: %3d \t return: %.3f \t ep_len: %.3f'%
                (epoch, np.mean(log_ep_rew), np.mean(log_ep_len)))
            # Reset logging info
            log_ep_len = []
            log_ep_rew = []
            if render and (epoch==1 or epoch%render_every==0):
                render_agent() 

        def render_agent():
            _obs = test_env.reset()
            _done = False
            while not _done:
                _act = get_policy_action(_obs, 0)
                _obs, _rew, _done, _ = test_env.step(_act)
                test_env.render()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--pi_lr', type=float, default=1e-3)
    parser.add_argument('--q_lr', type=float, default=1e-3)
    args = parser.parse_args()
    print('\nTwin Delayed Deep Deterministic Policy Gradient (TD3).\n')
    train(env_name=args.env_name, render=args.render, pi_lr=args.pi_lr, q_lr=args.q_lr)

