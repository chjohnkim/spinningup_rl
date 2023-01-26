import gym

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the skrl components to build the RL system
from skrl.models.torch import Model, DeterministicMixin
#from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch.wrappers import GymWrapper
from skrl.resources.preprocessors.torch import RunningStandardScaler
from fetch_wrappers import HERGoalEnvWrapper
from fetch_wrappers import HERRandomMemory, FetchRandomMemory
#from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from ddpg_custom import DDPG, DDPG_DEFAULT_CONFIG
from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise

from arguments import get_args
args = get_args()

if args.her:
    RandomMemory = HERRandomMemory
    print('Activating Hindsight Experience Replay')
else:
    RandomMemory = FetchRandomMemory
    print('Deactivating Hindsight Experience Replay')

# Define the models (stochastic and deterministic models) for the SAC agent using the mixins.
# - StochasticActor (policy): takes as input the environment's observation/state and returns an action
# - Critic: takes the state and action as input and provides a value to guide the policy
class Actor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, self.num_actions))

    def compute(self, states, taken_actions, role):
        return torch.tanh(self.net(states))
        
class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))

    def compute(self, states, taken_actions, role):        
        return self.net(torch.cat([states, taken_actions], dim=1))



# Load and wrap the Gym environment.
# Note: the environment version may change depending on the gym version
# Load and wrap the Gym environment.
# Note: the environment version may change depending on the gym version
env = gym.make(args.env_name)
env = HERGoalEnvWrapper(env)
env = GymWrapper(env)
device = env.device

# Instantiate a RandomMemory (without replacement) as experience replay memory
memory = RandomMemory(memory_size=args.replay_size, num_envs=env.num_envs, device=device, replacement=False, env=env)

# Instantiate the agent's models (function approximators).
# DDPG requires 4 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ddpg.html#spaces-and-models
models = {}
models["policy"] = Actor(env.observation_space, env.action_space, device, True)
models["target_policy"] = Actor(env.observation_space, env.action_space, device, True)
models["critic"] = Critic(env.observation_space, env.action_space, device)
models["target_critic"] = Critic(env.observation_space, env.action_space, device)

# Initialize the models' parameters (weights and biases) using a Gaussian distribution
for model in models.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ddpg.html#configuration-and-hyperparameters
cfg = DDPG_DEFAULT_CONFIG.copy()
cfg["experiment"]["write_interval"] = args.log_interval
if args.eval:
    cfg["experiment"]["checkpoint_interval"] = 0
    cfg["update_every"] = args.update_every
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
else:
    cfg["experiment"]["checkpoint_interval"] = args.checkpoint_interval
    cfg["gradient_steps"] = args.update_iters
    cfg["update_every"] = args.update_every
    cfg["batch_size"] = args.batch_size
    cfg["discount_factor"] = args.gamma
    cfg["polyak"] = 1 - args.polyak
    cfg["actor_learning_rate"] = args.pi_lr
    cfg["critic_learning_rate"] = args.q_lr
    cfg["learning_rate_scheduler"] = None
    cfg["learning_rate_scheduler_kwargs"] = {}

    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    #cfg["value_preprocessor"] = RunningStandardScaler
    #cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    
    cfg["random_timesteps"] = args.random_timesteps
    cfg["learning_starts"] = args.update_after
    # SAC
    cfg["learn_entropy"] = args.learn_entropy
    cfg["entropy_learning_rate"] = 1e-3
    cfg["initial_entropy_value"] = args.alpha
    cfg["target_entropy"] = None
    # DDPG
    cfg["exploration"]["noise"] = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.1, base_scale=1.0, device=device)
    cfg["exploration"]["initial_scale"] = 1.0
    cfg["exploration"]["final_scale"] = 1e-3
    cfg["exploration"]["timesteps"] = 5_000_000

agent_ddpg = DDPG(models=models,
                  memory=memory,
                  cfg=cfg,
                  observation_space=env.observation_space,
                  action_space=env.action_space,
                  device=device)
if args.eval:
    agent_ddpg.load("./runs/FetchSlide_DDPGHER_success/checkpoints/best_agent.pt")
    cfg_trainer = {"timesteps": 15000, "headless": (not args.render)}
else:
    # Configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": args.total_steps, "headless": (not args.render)}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent_ddpg)

if args.eval:
    trainer.eval()
else:
    # start training
    trainer.train()