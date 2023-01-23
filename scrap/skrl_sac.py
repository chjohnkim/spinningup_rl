import gym

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the skrl components to build the RL system
from skrl.models.torch import Model, DeterministicMixin, GaussianMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env

EVAL = False

# Define the models (stochastic and deterministic models) for the agents using mixins.
# - StochasticActor: takes as input the environment's observation/state and returns an action
# - Critic: takes the state and action as input and provides a value to guide the policy
class StochasticActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, states, taken_actions, role):
        return 2*torch.tanh(self.net(states)), self.log_std_parameter

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))

    def compute(self, states, taken_actions, role):
        return self.net(torch.cat([states, taken_actions], dim=1))

# Load and wrap the Gym environment.
# Note: the environment version may change depending on the gym version
try:
    env = gym.make("Pendulum-v1")
except gym.error.DeprecatedEnv as e:
    env_id = [spec.id for spec in gym.envs.registry.all() if spec.id.startswith("Pendulum-v")][0]
    print("Pendulum-v1 not found. Trying {}".format(env_id))
    env = gym.make(env_id)
env = wrap_env(env)

device = env.device


# Instantiate a RandomMemory (without replacement) as experience replay memory
memory = RandomMemory(memory_size=15000, num_envs=env.num_envs, device=device, replacement=False)

# SAC requires 5 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.sac.html#spaces-and-models
models_sac = {}
models_sac["policy"] = StochasticActor(env.observation_space, env.action_space, device, clip_actions=False)
models_sac["critic_1"] = Critic(env.observation_space, env.action_space, device)
models_sac["critic_2"] = Critic(env.observation_space, env.action_space, device)
models_sac["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
models_sac["target_critic_2"] = Critic(env.observation_space, env.action_space, device)

# Initialize the models' parameters (weights and biases) using a Gaussian distribution
for model in models_sac.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.sac.html#configuration-and-hyperparameters
cfg_sac = SAC_DEFAULT_CONFIG.copy()

# logging to TensorBoard and write checkpoints each 25 and 1000 timesteps respectively
cfg_sac["experiment"]["write_interval"] = 25

if EVAL:
    cfg_sac["experiment"]["checkpoint_interval"] = 0
else:
    cfg_sac["experiment"]["checkpoint_interval"] = 5000
    cfg_sac["batch_size"] = 100
    cfg_sac["learning_starts"] = 1000
    cfg_sac["learn_entropy"] = True

agent_sac = SAC(models=models_sac,
                memory=memory,
                cfg=cfg_sac,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)
if EVAL:
    agent_sac.load("./runs/23-01-12_15-54-20-011625_SAC/checkpoints/best_agent.pt")
    cfg_trainer = {"timesteps": 15000, "headless": False}
else:
    # Configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": 15000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent_sac)

if EVAL:
    trainer.eval()
else:
    # start training
    trainer.train()