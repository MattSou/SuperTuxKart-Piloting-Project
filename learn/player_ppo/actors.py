import collections.abc
import gymnasium as gym
from bbrl.agents import Agent
import torch
import collections
from gymnasium.spaces.dict import Dict
from torch import nn
from torch.distributions import Normal
from bbrl_utils.nn import build_mlp, build_ortho_mlp
import numpy as np
from torch.distributions import (
    Normal,
    Independent,
    TransformedDistribution,
    TanhTransform,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

class VAgent(Agent):
    def __init__(self, state_dim, hidden_layers, name="critic"):
        super().__init__(name)
        self.is_q_function = False
        self.model = build_ortho_mlp(
            [state_dim] + list(hidden_layers) + [1], activation=nn.ReLU()
        )

    def forward(self, t, **kwargs):
        X_velocity = self.get(("env/env_obs/velocity", t))
        X_center_path_distance = self.get(("env/env_obs/center_path_distance", t))
        X_center_path = self.get(("env/env_obs/center_path", t))
        X_front = self.get(("env/env_obs/front", t))
        X_paths_start = self.get(("env/env_obs/paths_start", t)).permute(1, 0, 2)[0]
        X_paths_end = self.get(("env/env_obs/paths_end", t)).permute(1, 0, 2)[0]
        observation = torch.cat([X_velocity, X_center_path_distance, X_center_path, X_front, X_paths_start, X_paths_end], dim=-1)

        critic = self.model(observation).squeeze(-1)
        self.set((f"{self.prefix}v_values", t), critic)

class DiscretePolicy(Agent):
    """
    Agent to predict steer based on constant maximum acceleration
    """
    def __init__(self, state_dim, hidden_size, n_actions, name="policy"):
        super().__init__(name=name)
        self.model = build_ortho_mlp(
            [state_dim] + list(hidden_size) + [n_actions], activation=nn.ReLU()
        )

    def dist(self, obs):
        scores = self.model(obs)
        probs = torch.softmax(scores, dim=-1)
        return torch.distributions.Categorical(probs)

    def forward(
        self,
        t,
        *,
        stochastic=True,
        predict_proba=False,
        compute_entropy=False,
        **kwargs,
    ):
        """
        Compute the action given either a time step (looking into the workspace)
        or an observation (in kwargs)
        """
        X_velocity = self.get(("env/env_obs/velocity", t))
        X_center_path_distance = self.get(("env/env_obs/center_path_distance", t))
        X_center_path = self.get(("env/env_obs/center_path", t))
        X_front = self.get(("env/env_obs/front", t))
        X_paths_start = self.get(("env/env_obs/paths_start", t)).permute(1, 0, 2)[0]
        X_paths_end = self.get(("env/env_obs/paths_end", t)).permute(1, 0, 2)[0]
        observation = torch.cat([X_velocity, X_center_path_distance, X_center_path, X_front, X_paths_start, X_paths_end], dim=-1)

        scores = self.model(observation)
        probs = torch.softmax(scores, dim=-1)
        if predict_proba:
            action = self.get(("action/steer", t))
            log_probs = probs[torch.arange(probs.size()[0]), action].log()
            self.set((f"{self.prefix}logprob_predict", t), log_probs)
        else:
            if stochastic:
                action = torch.distributions.Categorical(probs).sample()
            else:
                action = scores.argmax(1)
            self.set(("action/acceleration", t), torch.tensor(4).unsqueeze(0))
            self.set(("action/steer", t), action)

        if compute_entropy:
            entropy = torch.distributions.Categorical(probs).entropy()
            self.set((f"{self.prefix}entropy", t), entropy)
            

class DoubleDiscretePolicy(Agent):
    """
    Agent to predict acceleration AND steer actions
    """
    def __init__(self, state_dim, hidden_size, n_actions, name="policy"):
        super().__init__(name=name)
        self.init = build_ortho_mlp(
            [state_dim] + list(hidden_size) + [n_actions], activation=nn.ReLU()
        )
        #self.init.load_state_dict(torch.load("marche/actor_20250215184429.pth"))

        # Récupération des couches sous forme de liste
        layers = list(self.init.children())
        # Séparation du modèle
        self.model = nn.Sequential(*layers[:-2])  # Toutes les couches sauf la dernière
        self.model_steer = nn.Sequential(layers[-2])  # Dernière couche
        self.model_acceleration = nn.Linear(hidden_size[-1], 5)

    def dist(self, obs):
        scores = self.model(obs)
        probs = torch.softmax(scores, dim=-1)
        return torch.distributions.Categorical(probs)

    def forward(
        self,
        t,
        *,
        stochastic=True,
        predict_proba=False,
        compute_entropy=False,
        **kwargs,
    ):
        """
        Compute the action given either a time step (looking into the workspace)
        or an observation (in kwargs)
        """
        X_velocity = self.get(("env/env_obs/velocity", t))
        X_center_path_distance = self.get(("env/env_obs/center_path_distance", t))
        X_center_path = self.get(("env/env_obs/center_path", t))
        X_front = self.get(("env/env_obs/front", t))
        X_paths_start = self.get(("env/env_obs/paths_start", t)).permute(1, 0, 2)[0]
        X_paths_end = self.get(("env/env_obs/paths_end", t)).permute(1, 0, 2)[0]
        observation = torch.cat([X_velocity, X_center_path_distance, X_center_path, X_front, X_paths_start, X_paths_end], dim=-1)

        hidden = self.model(observation)
        scores_steer= self.model_steer(hidden)
        scores_acceleration = self.model_acceleration(hidden)

        probs_steer = torch.softmax(scores_steer, dim=-1)
        probs_acceleration = torch.softmax(scores_acceleration, dim=-1)
        if predict_proba:
            action_steer = self.get(("action/steer", t))
            action_acceleration = self.get(("action/acceleration", t))
            log_probs_steer = probs_steer[torch.arange(probs_steer.size()[0]), action_steer].log()
            log_probs_acceleration = probs_acceleration[torch.arange(probs_acceleration.size()[0]), action_acceleration].log()
            self.set((f"{self.prefix}logprob_predict", t), log_probs_steer + log_probs_acceleration)
        else:
            if stochastic:
                action_steer = torch.distributions.Categorical(probs_steer).sample()
                action_acceleration = torch.distributions.Categorical(probs_acceleration).sample()
            else:
                action_steer = scores_steer.argmax(1)
                action_acceleration = scores_acceleration.argmax(1)
                
            self.set(("action/steer", t), action_steer)
            self.set(("action/acceleration", t), action_acceleration)

        if compute_entropy:
            entropy_steer = torch.distributions.Categorical(probs_steer).entropy()
            entropy_acceleration = torch.distributions.Categorical(probs_acceleration).entropy()
            entropy = entropy_steer + entropy_acceleration
            self.set((f"{self.prefix}entropy", t), entropy)