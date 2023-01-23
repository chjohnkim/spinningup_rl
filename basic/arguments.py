import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # SAC, TD3, DDPG
    parser.add_argument('--env_name', '--env', type=str, default='FetchReach-v1') # Name of the environment
    parser.add_argument('--hidden_sizes', type=list, default=[256, 256]) # Hidden layer sizes
    parser.add_argument('--replay_size', type=int, default=int(1e6)) # Size of the replay buffer
    parser.add_argument('--epochs', type=int, default=50) # Number of epochs to train for
    parser.add_argument('--total_steps', type=int, default=10_000_000) # Total number of steps to train for
    parser.add_argument('--batch_size', type=int, default=256) # Batch size for training
    parser.add_argument('--steps_per_epoch', type=int, default=4_000) # Number of steps to collect per epoch
    parser.add_argument('--start_steps', type=int, default=10_000) 
    parser.add_argument('--random_timesteps', type=int, default=5_000) # Number of steps for uniform-random action selection
    parser.add_argument('--update_after', type=int, default=5_000) # Number of steps to collect before starting training
    parser.add_argument('--update_every', type=int, default=200) # Number of steps to collect before updating
    parser.add_argument('--update_iters', type=int, default=40) # Number of updates to perform per update cycle
    parser.add_argument('--pi_lr', type=float, default=1e-3) # Learning rate for policy
    parser.add_argument('--q_lr', type=float, default=1e-3) # Learning rate for Q
    parser.add_argument('--gamma', type=float, default=0.99) # Discount factor
    parser.add_argument('--polyak', type=float, default=0.995) # Polyak averaging coefficient
    # SAC
    parser.add_argument('--alpha', type=float, default=0.2) # Entropy regularization coefficient
    parser.add_argument('--learn_entropy', action='store_true') # Whether to learn the entropy coefficient

    # TD3
    parser.add_argument('--target_noise', type=float, default=0.2) # Noise added to target policy
    parser.add_argument('--noise_clip', type=float, default=0.5) # Range to clip target policy noise
    parser.add_argument('--policy_delay', type=int, default=2) # Policy update frequency
    # DDPG & TD3
    parser.add_argument('--act_noise', type=float, default=0.1) # Std(?) of Gaussian exploration noise added to policy
    # HER
    parser.add_argument('--her', action='store_true') # Whether to use HER
    parser.add_argument('--her_k', type=int, default=4) # Number of HER transitions to sample per regular transition
    parser.add_argument('--her_goal_selection_strategy', type=str, default='future') # HER goal selection strategy

    # Util
    parser.add_argument('--eval', action='store_true') # Whether to evaluate the policy
    parser.add_argument('--log_dir', type=str, default='experiments') # Directory to save logs to
    parser.add_argument('--render_every', type=int, default=1) # How often to render the environment
    parser.add_argument('--render', action='store_true') # Whether to render the environment
    parser.add_argument('--checkpoint_interval', type=int, default=2000) # How often to save the model
    parser.add_argument('--log_interval', type=int, default=1000) # How often to log training progress
    
    args = parser.parse_args()
    return args