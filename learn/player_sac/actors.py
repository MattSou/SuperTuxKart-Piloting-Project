import collections.abc
import gymnasium as gym
from bbrl.agents import Agent
import torch
import collections
from gymnasium.spaces.dict import Dict
from torch import nn
from torch.distributions import Normal
from bbrl_utils.nn import build_mlp
import numpy as np
from torch.distributions import (
    Normal,
    Independent,
    TransformedDistribution,
    TanhTransform,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   


class SquashedGaussianActor(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim, min_std=1e-4):
        """Creates a new Squashed Gaussian actor

        :param state_dim: The dimension of the state space
        :param hidden_layers: Hidden layer sizes
        :param action_dim: The dimension of the action space
        :param min_std: The minimum standard deviation, defaults to 1e-4
        """
        super().__init__()
        self.min_std = min_std
        backbone_dim = [state_dim] + list(hidden_layers)
        self.layers = build_mlp(backbone_dim, activation=nn.ReLU())
        self.backbone = nn.Sequential(*self.layers)
        self.last_mean_layer = nn.Linear(hidden_layers[-1], action_dim)
        self.last_std_layer = nn.Linear(hidden_layers[-1], action_dim)
        self.softplus = nn.Softplus()
        
        # cache_size avoids numerical infinites or NaNs when
        # computing log probabilities
        self.tanh_transform = TanhTransform(cache_size=1)

    def normal_dist(self, obs: torch.Tensor):
        """Compute normal distribution given observation(s)"""
        
        backbone_output = self.backbone(obs)
        mean = self.last_mean_layer(backbone_output)
        std_out = self.last_std_layer(backbone_output)
        std = self.softplus(std_out) + self.min_std
        # Independent ensures that we have a multivariate
        # Gaussian with a diagonal covariance matrix (given as
        # a vector `std`)
        return Independent(Normal(mean, std), 1)

    def forward(self, t, stochastic=True):
        """Computes the action a_t and its log-probability p(a_t| s_t)

        :param stochastic: True when sampling
        """
        continuous_obs = self.get(("env/env_obs/continuous", t))
        discrete_obs = self.get(("env/env_obs/discrete", t))
        obs = torch.cat((continuous_obs, discrete_obs), dim=1)
        normal_dist = self.normal_dist(obs)
        action_dist = TransformedDistribution(normal_dist, [self.tanh_transform])
        if stochastic:
            # Uses the re-parametrization trick
            action = action_dist.rsample()
        else:
            # Directly uses the mode of the distribution
            action = self.tanh_transform(normal_dist.mode)

        log_prob = action_dist.log_prob(action)
        # This line allows to deepcopy the actor...
        self.tanh_transform._cached_x_y = [None, None]
        self.set(("action", t), action)
        self.set(("action_logprobs", t), log_prob)
class ContinuousQAgent(Agent):
    def __init__(self, state_dim: int, hidden_layers: list[int], action_dim: int):
        """Creates a new critic agent $Q(s, a)$

        :param state_dim: The number of dimensions for the observations
        :param hidden_layers: The list of hidden layers for the NN
        :param action_dim: The numer of dimensions for actions
        """
        super().__init__()
        self.is_q_function = True
        self.model = build_mlp(
            [state_dim + action_dim] + list(hidden_layers) + [1], activation=nn.ReLU()
        )

    def forward(self, t):
        continuous_obs = self.get(("env/env_obs/continuous", t))
        discrete_obs = self.get(("env/env_obs/discrete", t))
        obs = torch.cat((continuous_obs, discrete_obs), dim=1)
        action = self.get(("action", t))
        obs_act = torch.cat((obs, action), dim=1)
        q_value = self.model(obs_act).squeeze(-1)
        self.set((f"{self.prefix}q_value", t), q_value)