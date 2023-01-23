import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
import gym
from torch.distributions.normal import Normal
from gym.spaces import Box
from copy import deepcopy
import core 
from her import HindsightExperienceReplayWrapper
from logger import Logger

class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, 
                 act_limit, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.net = core.mlp([obs_dim]+hidden_sizes, activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], 1)
        self.act_limit = act_limit
        self.LOG_STD_MIN = log_std_min
        self.LOG_STD_MAX = log_std_max

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used at test time
            pi_action = mu
        else:
            # Stochastic policy 
            pi_action = pi_distribution.rsample()
        
        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1) 
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit*pi_action
        return pi_action, logp_pi

class ActorCritic:
    def __init__(self, obs_dim, act_dim, act_limit, hidden_sizes=[256, 256, 256]):
        self.pi = SquashedGaussianActor(obs_dim, act_dim, hidden_sizes, nn.ReLU, act_limit)
        self.q1 = core.mlp([obs_dim+act_dim]+hidden_sizes+[1])
        self.q2 = core.mlp([obs_dim+act_dim]+hidden_sizes+[1])
        
    def act(self, obs, deterministic=False): # TODO: return logp as well. Also need deterministic or not
        with torch.no_grad():
            action, _ = self.pi(obs, deterministic, False)
        return action.numpy()

    def get_q1(self, obs, action):
        q = self.q1(torch.cat([obs, action], dim=-1)) 
        return torch.squeeze(q, -1) 

    def get_q2(self, obs, action):
        q = self.q2(torch.cat([obs, action], dim=-1)) 
        return torch.squeeze(q, -1) 

def train(env_name='FetchReach-v1', hidden_sizes=[256, 256], replay_size=int(1e6), 
          epochs=500, batch_size=256, steps_per_epoch=4_000, start_steps=10_000,
          update_after=1_000, update_every=50, update_iters=50,   
          pi_lr=1e-3, q_lr=1e-3, gamma=0.99, polyak=0.95, alpha=0.2, 
          her=False, her_k=4, her_goal_selection_strategy='future',
          log_dir='experiments', render_every=1, render=False):
    if her:
        alg_name = 'sac_her'
    else:
        alg_name = 'sac'
    logger = Logger(log_dir, env_name, alg_name) 
    env = gym.make(env_name)
    test_env = gym.make(env_name)

    assert isinstance(env.observation_space['observation'], Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Box), \
        "This example only works for envs with continuous action spaces."

    obs_dim = env.observation_space['observation'].shape[0] + env.observation_space['desired_goal'].shape[0]
    act_dim = env.action_space.shape[0]
    max_ep_len = env._max_episode_steps
    # Action limit for clamping, critically, assumes all dimensions share the same bound
    act_limit = env.action_space.high[0]

    # Create actor critic model and target networks
    ac = ActorCritic(obs_dim, act_dim, act_limit, hidden_sizes)
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
    # Wrap replay buffer if using Hindsight Experience Replay
    if her:
        replay_buffer = HindsightExperienceReplayWrapper(replay_buffer=replay_buffer, 
                                                         n_sampled_goal=her_k, 
                                                         goal_selection_strategy=her_goal_selection_strategy,
                                                         env=gym.make(env_name)
                                                        )

    ''' KEY PART: get_policy_action(), compute_loss_q(), compute_loss_pi(), ActorCritic'''
    ''' Parameter for deterministic or stochastic'''
    def get_policy_action(obs, deterministic=False):
        return ac.act(torch.as_tensor(obs, dtype=torch.float32), deterministic)

    # Function for computing SAC Q-loss
    def compute_loss_q(data):
        obs = data['obs']
        action = data['act']
        reward = data['rew']
        next_obs = data['next_obs']
        done = data['done']
        
        q1 = ac.get_q1(obs, action) 
        q2 = ac.get_q2(obs, action)

        # Bellman backup for Q function
        with torch.no_grad():
            ''' Target action comes from *current* policy '''
            next_target_action, logp_next_target_action = ac.pi(next_obs)

            q1_pi_target = ac_target.get_q1(next_obs, next_target_action)
            q2_pi_target = ac_target.get_q2(next_obs, next_target_action)
            q_pi_target = torch.min(q1_pi_target, q2_pi_target)
            backup = reward + gamma * (1-done) * (q_pi_target - alpha*logp_next_target_action)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2
        return loss_q

    # Function for computing SAC pi loss 
    ''' Implements entropy regularized policy loss '''
    def compute_loss_pi(data): 
        obs = data['obs']
        pi, logp_pi = ac.pi(obs) 
        q1_pi = ac.get_q1(obs, pi)
        q2_pi = ac.get_q2(obs, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        ''' entropy regularization '''
        loss_pi = (alpha * logp_pi - q_pi).mean()  
        return loss_pi


    ''' 
    The only difference from TD3 is that less frequent updates on policy and targets are removed.
    '''
    def update(data):
        # First run one gradient step for Q
        q_optimizer.zero_grad()
        loss_q = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

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
    obs_dict = env.reset()
    obs = np.concatenate((obs_dict['observation'], obs_dict['desired_goal']))
    done = False
    ep_reward = 0
    ep_len = 0
    # Main loop: Collect experience in environment and update policy
    total_steps = steps_per_epoch * epochs
    for step in range(total_steps):
        # Until start steps have elapsed, randomly sample actions 
        # from uniform distribution for better exploration. 
        # Afterwards used learned policy with stochastic policy
        if step>start_steps:
            action = get_policy_action(obs, deterministic=False) 
        else:
            action = env.action_space.sample()

        # Step the environment
        next_obs_dict, reward, done, _ = env.step(action)
        # Because fetch env always returns done = False
        if reward==0:
            done = True

        next_obs = np.concatenate((next_obs_dict['observation'], next_obs_dict['desired_goal']))
        ep_len+=1
        ep_reward+=reward
        # Ignore done signal if it comes from max time horizon of episode
        done = False if ep_len==max_ep_len else done 

        # Store trajectory into replay buffer            
        if her:
            replay_buffer.store(obs, action, reward, next_obs, done, next_obs_dict['achieved_goal'])
        else:
            replay_buffer.store(obs, action, reward, next_obs, done)
        obs = next_obs

        # Handle end of episode or reached trajectory max len 
        if done or ep_len==max_ep_len:
            if her:
                replay_buffer.sample_her_transitions()
            log_ep_len.append(ep_len)
            log_ep_rew.append(ep_reward)
            # reset episode-specific variables
            obs_dict = env.reset()
            obs = np.concatenate((obs_dict['observation'], obs_dict['desired_goal']))
            done = False
            ep_len = 0
            ep_reward = 0       

        # Update every interval after enough steps have been taken
        if step>=update_after and step%update_every==0:
            for iter in range(update_iters): 
                batch_dict = replay_buffer.sample_batch(batch_size)
                update(batch_dict)
      
        # Epoch handling
        if (step+1)%steps_per_epoch==0:
            epoch = (step+1) // steps_per_epoch 
            logger.log(epoch, step+1, np.mean(log_ep_rew), np.mean(log_ep_len))
            # Reset logging info
            log_ep_len = []
            log_ep_rew = []
            if render and (epoch==1 or epoch%render_every==0):
                render_agent() 

        def render_agent():
            _obs_dict = test_env.reset()
            _obs = np.concatenate((_obs_dict['observation'], _obs_dict['desired_goal']))
            _done = False
            while not _done:
                _act = get_policy_action(_obs, deterministic=True)
                _obs_dict, _rew, _done, _ = test_env.step(_act)
                if _rew==0:
                    _done = True
                _obs = np.concatenate((_obs_dict['observation'], _obs_dict['desired_goal']))
                test_env.render()


if __name__ == '__main__':
    from arguments import get_args
    args = get_args()
    print('\nSoft Actor Critic.\n')
    train(env_name=args.env_name, 
          hidden_sizes=args.hidden_sizes, 
          replay_size=args.replay_size,
          epochs=args.epochs, 
          batch_size=args.batch_size, 
          steps_per_epoch=args.steps_per_epoch,
          start_steps=args.start_steps, 
          update_after=args.update_after, 
          update_every=args.update_every,
          update_iters=args.update_iters, 
          pi_lr=args.pi_lr, 
          q_lr=args.q_lr, 
          gamma=args.gamma,
          polyak=args.polyak, 
          alpha=args.alpha, 
          her=args.her, 
          her_k=args.her_k,
          her_goal_selection_strategy=args.her_goal_selection_strategy, 
          log_dir=args.log_dir,
          render_every=args.render_every, 
          render=args.render)
