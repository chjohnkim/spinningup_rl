from typing import Union, Tuple, Any, Optional, Sequence, List
from collections import OrderedDict
from gym import spaces
import numpy as np
from enum import Enum
import copy
import torch
from torch.distributions import Normal

from skrl.models.torch import GaussianMixin
from skrl.memories.torch import RandomMemory, Memory

class GoalSelectionStrategy(Enum):
    """
    The strategies for selecting new goals when
    creating artificial transitions.
    """
    # Select a goal that was achieved
    # after the current step, in the same episode
    FUTURE = 0
    # Select the goal that was achieved
    # at the end of the episode
    FINAL = 1
    # Select a goal that was achieved in the episode
    EPISODE = 2
    # Select a goal that was achieved
    # at some point in the training procedure
    # (and that is present in the replay buffer)
    RANDOM = 3

# For convenience
# that way, we can use string to select a strategy
KEY_TO_GOAL_STRATEGY = {
    'future': GoalSelectionStrategy.FUTURE,
    'final': GoalSelectionStrategy.FINAL,
    'episode': GoalSelectionStrategy.EPISODE,
    'random': GoalSelectionStrategy.RANDOM
}


class HERRandomMemory(Memory):
    def __init__(self, 
                 memory_size: int, 
                 num_envs: int = 1, 
                 device: Union[str, torch.device] = "cuda:0", 
                 export: bool = False, 
                 export_format: str = "pt", 
                 export_directory: str = "", 
                 replacement=True,
                 n_sampled_goal: int=4,
                 goal_selection_strategy: str='future',
                 env=None) -> None:
        super().__init__(memory_size, num_envs, device, export, export_format, export_directory)

        self._replacement = replacement

        self.n_sampled_goal = n_sampled_goal
        if type(goal_selection_strategy) is str:
            goal_selection_strategy = KEY_TO_GOAL_STRATEGY[goal_selection_strategy]
        self.goal_selection_strategy = goal_selection_strategy
        self.env = env
        self.max_episode_steps = env._max_episode_steps
        self.future_p = 1 - (1. / (1 + n_sampled_goal))

        # Buffer for storing transitions of the current episode
        self.episode_transitions = []

    def add_samples(self, **tensors: torch.Tensor) -> None:
        # Add the transition to the current episode buffer
        t = copy.deepcopy(tensors)
        self.episode_transitions.append(t)
        # if the episode is over, add the transitions to the replay buffer
        # and also sample imagined transitions based on HER sample strategy
        goal_reached = t['dones'][0] #t['rewards'][0] == 0
        if goal_reached:
            # Add the transitions of the current episode to the replay buffer
            self._add_episode_transitions()
            # Reset the episode buffer
            self.episode_transitions = []

    def _add_episode_transitions(self):
        '''
        Sample artificial goals and store transition of the current episode in replay buffer.
        This method should be called only after each end of episode. 
        '''
        # Add the transitions of the current episode to the replay buffer
        for transition in self.episode_transitions:
            # Add the transition to the replay buffer
            super().add_samples(**transition)

    def sample(self, names: Tuple[str], batch_size: int, mini_batches: int = 1) -> List[List[torch.Tensor]]:        
        if self.filled:
            mem_size = self.memory_size
        else:
            mem_size = self.memory_index
        num_eps = mem_size//self.max_episode_steps
        tensors_copy = copy.deepcopy(self.tensors)
        tensors_copy = {name: tensors_copy[name][:mem_size] for name in names} 
        
        states = self.env.convert_obs_to_dict_vector(tensors_copy['states'].squeeze(1))
        next_states = self.env.convert_obs_to_dict_vector(tensors_copy['next_states'].squeeze(1))

        episode_batch = dict(
            obs=states['observation'].reshape(num_eps, self.max_episode_steps, -1),
            ag=states['achieved_goal'].reshape(num_eps, self.max_episode_steps, -1),
            g=states['desired_goal'].reshape(num_eps, self.max_episode_steps, -1),
            r=tensors_copy['rewards'].reshape(num_eps, self.max_episode_steps, -1),
            actions=tensors_copy['actions'].reshape(num_eps, self.max_episode_steps, -1),
            obs_next=next_states['observation'].reshape(num_eps, self.max_episode_steps, -1),
            ag_next=next_states['achieved_goal'].reshape(num_eps, self.max_episode_steps, -1),
            dones=tensors_copy['dones'].reshape(num_eps, self.max_episode_steps, -1),
        )

        her_transitions = self.sample_her_transitions(episode_batch, batch_size)
        
        obs_dict = dict(
            observation=her_transitions['obs'],
            desired_goal=her_transitions['g'],
            achieved_goal=her_transitions['ag'],
        )
        obs = self.env.convert_batch_dict_to_obs_tensor(obs_dict)
        obs_dict = dict(
            observation=her_transitions['obs_next'],
            desired_goal=her_transitions['g'],
            achieved_goal=her_transitions['ag_next'],
        )
        obs_next = self.env.convert_batch_dict_to_obs_tensor(obs_dict)

        return (obs, her_transitions['actions'], her_transitions['r'], obs_next, her_transitions['dones']), None


    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):        
        T = episode_batch['actions'].shape[1]-1
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size) # Which episodes [0,rollout_nums]
        t_samples = np.random.randint(T, size=batch_size) # Which timesteps in episode [0,50]
        transitions = {key: episode_batch[key][episode_idxs, t_samples] for key in episode_batch.keys()} # Selected random transitions
        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p) # From the selected transitions, which ones to be replaced with future goals
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int) # Random amount of timesteps to be added to the selected timestep
        future_t = (t_samples + 1 + future_offset)[her_indexes] # The timestep to be replaced with future goal
        # replace go with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag
        # to get the params to re-compute reward
        transitions['r'] = torch.tensor(np.expand_dims(self.env.compute_reward(transitions['ag_next'], transitions['g'], None), 1), device=self.device)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}
        return transitions

class HERRandomMemory_stablebaselines(RandomMemory):
    def __init__(self, 
                 memory_size: int, 
                 num_envs: int = 1, 
                 device: Union[str, torch.device] = "cuda:0", 
                 export: bool = False, 
                 export_format: str = "pt", 
                 export_directory: str = "", 
                 replacement: bool = True,
                 n_sampled_goal: int=4,
                 goal_selection_strategy: str='future',
                 env=None) -> None:
        super().__init__(memory_size, num_envs, device, export, export_format, export_directory, replacement)


        self.n_sampled_goal = n_sampled_goal
        if type(goal_selection_strategy) is str:
            goal_selection_strategy = KEY_TO_GOAL_STRATEGY[goal_selection_strategy]
        self.goal_selection_strategy = goal_selection_strategy
        self.env = env
        self.max_episode_steps = env._max_episode_steps

        #self.goal_dim = self.env.observation_space['desired_goal'].shape[0] # TODO: Check if this is needed
        # Buffer for storing transitions of the current episode
        self.episode_transitions = []

    def add_samples(self, **tensors: torch.Tensor) -> None:
        # Add the transition to the current episode buffer
        t = copy.deepcopy(tensors)
        self.episode_transitions.append(t)
        # if the episode is over, add the transitions to the replay buffer
        # and also sample imagined transitions based on HER sample strategy
        goal_reached = t['dones'][0] #t['rewards'][0] == 0
        if goal_reached:
            # Add the transitions of the current episode to the replay buffer
            self._add_episode_transitions()
            # Reset the episode buffer
            self.episode_transitions = []

    def _sample_achieved_goal(self, episode_transitions, transition_idx):
        '''
        Sample an achieved goal according to sampling strategy
        '''
        if self.goal_selection_strategy==GoalSelectionStrategy.FUTURE:
            # Sample a goal that was observed in the same episode after the current step
            selected_idx = np.random.choice(np.arange(transition_idx+1, len(episode_transitions)))
            selected_transition = episode_transitions[selected_idx]
        
        elif self.goal_selection_strategy==GoalSelectionStrategy.FINAL:
            # Choose the goal achieved at the end of the episode
            selected_transition = episode_transitions[-1]
        
        elif self.goal_selection_strategy==GoalSelectionStrategy.EPISODE:
            # Random goal achieved during the episode
            selected_idx = np.random.chocie(np.arange(len(episode_transitions)))
            selected_transition = episode_transitions[selected_idx]

        elif self.goal_selection_strategy==GoalSelectionStrategy.RANDOM:
            raise NotImplementedError("random goal selection strategy not implemented yet!")
        else:
            raise ValueError("Invalid goal selection strategy,"
                             "please use one of {}".format(list(GoalSelectionStrategy)))
        
        # TODO: Check if squeeze and unsqueeze is needed
        next_state_dict = self.env.convert_obs_to_dict(selected_transition['next_states'].squeeze())
        next_achieved_goal = next_state_dict['achieved_goal'].unsqueeze(0) 
        return next_achieved_goal 

    def _add_episode_transitions(self):
        '''
        Sample artificial goals and store transition of the current episode in replay buffer.
        This method should be called only after each end of episode. 
        '''
        # Add the transitions of the current episode to the replay buffer
        ep_trans = copy.deepcopy(self.episode_transitions)
        for transition_idx, transition in enumerate(ep_trans):
            # Add the transition to the replay buffer
            t = copy.deepcopy(transition)
            super().add_samples(**t)
            
            # We cannot sample a goal from the future in the last step of an episode
            if (transition_idx==len(ep_trans)-1 and 
                self.goal_selection_strategy==GoalSelectionStrategy.FUTURE):
                break
            
            # Sampled n goals per transition, where n is 'n_sampled_goal' (param k in paper)
            artificial_goals = [
                self._sample_achieved_goal(ep_trans, transition_idx)
                for _ in range(self.n_sampled_goal)
            ]

            # For each sampled goals, store a new transition
            for artificial_goal in artificial_goals:
                # Copy transition to avoid modifying the original one
                artificial_transition = copy.deepcopy(transition)

                # Update the desired goal in the transtion with the sampled goal
                # states
                states_dict = self.env.convert_obs_to_dict(artificial_transition['states'].squeeze())
                states_dict['desired_goal'] = artificial_goal.squeeze()
                artificial_transition['states'] = self.env.convert_dict_to_obs_tensor(states_dict).unsqueeze(0)
                # next_states
                next_states_dict = self.env.convert_obs_to_dict(artificial_transition['next_states'].squeeze())
                next_states_dict['desired_goal'] = artificial_goal.squeeze()
                artificial_transition['next_states'] = self.env.convert_dict_to_obs_tensor(next_states_dict).unsqueeze(0)

                # Update the reward according to the new desired goal
                rew = self.env.compute_reward(next_states_dict['achieved_goal'].unsqueeze(0), artificial_goal, None)
                artificial_transition['rewards'] = torch.tensor(rew, device=self.device).unsqueeze(0)
                if rew==0:
                    artificial_transition['dones'] = torch.tensor([True], device=self.device).unsqueeze(0)
                else:
                    artificial_transition['dones'] = torch.tensor([False], device=self.device).unsqueeze(0)            
                # Add the transition to the replay buffer
                super().add_samples(**artificial_transition) # TODO
            
class FetchRandomMemory(RandomMemory):
    def __init__(self, 
                 memory_size: int, 
                 num_envs: int = 1, 
                 device: Union[str, torch.device] = "cuda:0", 
                 export: bool = False, 
                 export_format: str = "pt", 
                 export_directory: str = "", 
                 replacement=True,
                 env=None) -> None:
        super().__init__(memory_size, num_envs, device, export, export_format, export_directory, replacement)

    def add_samples(self, **tensors: torch.Tensor) -> None: 
        super().add_samples(**tensors)


class FetchGaussianMixin(GaussianMixin):
    def __init__(self, 
                 clip_actions: bool = False, 
                 clip_log_std: bool = True, 
                 min_log_std: float = -20, 
                 max_log_std: float = 2,
                 reduction: str = "sum",
                 role: str = "") -> None:
        super().__init__(clip_actions, clip_log_std, min_log_std, max_log_std, reduction, role)
    
    # Overwriting method from parent class GaussianMixin
    def act(self, 
            states: torch.Tensor, 
            taken_actions: Optional[torch.Tensor] = None, 
            role: str = "") -> Sequence[torch.Tensor]:

        # map from states/observations to mean actions and log standard deviations
        actions_mean, log_std = self.compute(states.to(self.device), 
                                             taken_actions.to(self.device) if taken_actions is not None else taken_actions, role)

        # clamp log standard deviations
        if self._g_clip_log_std[role] if role in self._g_clip_log_std else self._g_clip_log_std[""]:
            log_std = torch.clamp(log_std, 
                                  self._g_log_std_min[role] if role in self._g_log_std_min else self._g_log_std_min[""],
                                  self._g_log_std_max[role] if role in self._g_log_std_max else self._g_log_std_max[""])

        self._g_log_std[role] = log_std
        self._g_num_samples[role] = actions_mean.shape[0]

        # distribution
        self._g_distribution[role] = Normal(actions_mean, log_std.exp())

        # sample using the reparameterization trick
        actions = self._g_distribution[role].rsample()

        # log of the probability density function
        log_prob = self._g_distribution[role].log_prob(actions if taken_actions is None else taken_actions)
        reduction = self._g_reduction[role] if role in self._g_reduction else self._g_reduction[""]
        if reduction is not None:
            log_prob = reduction(log_prob, dim=-1)
        log_prob -= (2*(torch.log(torch.tensor(2)) - actions - torch.nn.functional.softplus(-2*actions))).sum(axis=1) 
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)
       
        actions = torch.tanh(actions)
        '''
        # clip actions
        if self._g_clip_actions[role] if role in self._g_clip_actions else self._g_clip_actions[""]:
            if self._backward_compatibility:
                actions = torch.max(torch.min(actions, self.clip_actions_max), self.clip_actions_min)
            else:
                actions = torch.clamp(actions, min=self.clip_actions_min, max=self.clip_actions_max)
        '''
        actions = self.clip_actions_max * actions
        return actions, log_prob, actions_mean

# Important: gym mixes up ordered and unordered keys
# and the Dict space may return a different order of keys that the actual one
# https://stable-baselines.readthedocs.io/en/master/_modules/stable_baselines/her/utils.html#HERGoalEnvWrapper
KEY_ORDER = ['observation', 'achieved_goal', 'desired_goal']

class HERGoalEnvWrapper(object):
    """
    A wrapper that allow to use dict observation space (coming from GoalEnv) with
    the RL algorithms.
    It assumes that all the spaces of the dict space are of the same type.

    :param env: (gym.GoalEnv)
    """

    def __init__(self, env):
        super(HERGoalEnvWrapper, self).__init__()
        self.env = env
        self.metadata = self.env.metadata
        self.action_space = env.action_space
        self.spaces = list(env.observation_space.spaces.values())
        self._max_episode_steps = env._max_episode_steps
        # Check that all spaces are of the same type
        # (current limitation of the wrapper)
        space_types = [type(env.observation_space.spaces[key]) for key in KEY_ORDER]
        assert len(set(space_types)) == 1, "The spaces for goal and observation"\
                                           " must be of the same type"

        if isinstance(self.spaces[0], spaces.Discrete):
            self.obs_dim = 1
            self.goal_dim = 1
        else:
            goal_space_shape = env.observation_space.spaces['achieved_goal'].shape
            self.obs_dim = env.observation_space.spaces['observation'].shape[0]
            self.goal_dim = goal_space_shape[0]

            if len(goal_space_shape) == 2:
                assert goal_space_shape[1] == 1, "Only 1D observation spaces are supported yet"
            else:
                assert len(goal_space_shape) == 1, "Only 1D observation spaces are supported yet"

        if isinstance(self.spaces[0], spaces.MultiBinary):
            total_dim = self.obs_dim + 2 * self.goal_dim
            self.observation_space = spaces.MultiBinary(total_dim)

        elif isinstance(self.spaces[0], spaces.Box):
            lows = np.concatenate([space.low for space in self.spaces])
            highs = np.concatenate([space.high for space in self.spaces])
            self.observation_space = spaces.Box(lows, highs, dtype=np.float32)

        elif isinstance(self.spaces[0], spaces.Discrete):
            dimensions = [env.observation_space.spaces[key].n for key in KEY_ORDER]
            self.observation_space = spaces.MultiDiscrete(dimensions)

        else:
            raise NotImplementedError("{} space is not supported".format(type(self.spaces[0])))

    def convert_dict_to_obs(self, obs_dict):
        """
        :param obs_dict: (dict<np.ndarray>)
        :return: (np.ndarray)
        """
        # Note: achieved goal is not removed from the observation
        # this is helpful to have a revertible transformation
        if isinstance(self.observation_space, spaces.MultiDiscrete):
            # Special case for multidiscrete
            return np.concatenate([[int(obs_dict[key])] for key in KEY_ORDER])
        return np.concatenate([obs_dict[key] for key in KEY_ORDER])

    def convert_dict_to_obs_tensor(self, obs_dict):
        """
        :param obs_dict: (dict<torch.Tensor>)
        :return: Tensor
        """
        return torch.cat([obs_dict[key] for key in KEY_ORDER])

    def convert_batch_dict_to_obs_tensor(self, obs_dict):
        """
        :param obs_dict: (dict<torch.Tensor>)
        :return: Tensor
        """
        return torch.cat([obs_dict[key] for key in KEY_ORDER], dim=1)


    def convert_obs_to_dict(self, observations):
        """
        Inverse operation of convert_dict_to_obs

        :param observations: (np.ndarray)
        :return: (OrderedDict<np.ndarray>)
        """
        return OrderedDict([
            ('observation', observations[:self.obs_dim]),
            ('achieved_goal', observations[self.obs_dim:self.obs_dim + self.goal_dim]),
            ('desired_goal', observations[self.obs_dim + self.goal_dim:]),
        ])

    def convert_obs_to_dict_vector(self, observations):
        """
        Inverse operation of convert_dict_to_obs

        :param observations: (np.ndarray)
        :return: (OrderedDict<np.ndarray>)
        """
        return OrderedDict([
            ('observation', observations[:, :self.obs_dim]),
            ('achieved_goal', observations[:, self.obs_dim:self.obs_dim + self.goal_dim]),
            ('desired_goal', observations[:, self.obs_dim + self.goal_dim:]),
        ])


    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.convert_dict_to_obs(obs), reward, done, info

    def seed(self, seed=None):
        return self.env.seed(seed)

    def reset(self):
        return self.convert_dict_to_obs(self.env.reset())

    def compute_reward(self, achieved_goal, desired_goal, info):
        achieved_goal = achieved_goal.detach().cpu().numpy()
        desired_goal = desired_goal.detach().cpu().numpy()
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()