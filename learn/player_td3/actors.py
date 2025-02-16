import torch.nn.functional as F
import math
from bbrl.agents import Agent
import torch
from torch import nn
from torch.distributions import Normal
from bbrl_utils.nn import build_mlp
import numpy as np
from torch.distributions import (
    Normal,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   


class Net_specifique(nn.Module):
    def __init__(self, size_max_steer_angle, size_velocity, size_center_path_distance, size_center_path, size_front, size_paths_start, size_paths_end, size_paths_width):
        super(Net_specifique, self).__init__()
        self.fc_velocity = nn.Linear(size_velocity, 32)
        self.fc_center_path_distance = nn.Linear(size_center_path_distance, 32)
        self.fc_center_path = nn.Linear(size_center_path, 32)
        self.fc_front = nn.Linear(size_front, 32)
        self.fc_paths_start = nn.Linear(size_paths_start, 32)
        self.fc_paths_end = nn.Linear(size_paths_end, 32)
        self.fc1 = nn.Linear(32*6, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 256)

        self.fc5 = nn.Linear(256, 1)


        self.fc10 = nn.Linear(256, 1)

    def forward(self, x_max_steer_angle, x_velocity, x_center_path_distance, x_center_path, x_front, x_paths_start, x_paths_end, x_paths_width):
        x_velocity = F.relu(self.fc_velocity(x_velocity))
        x_center_path_distance = F.relu(self.fc_center_path_distance(x_center_path_distance))
        x_center_path = F.relu(self.fc_center_path(x_center_path))
        x_front = F.relu(self.fc_front(x_front))
        x_paths_start = F.relu(self.fc_paths_start(x_paths_start))
        x_paths_end = F.relu(self.fc_paths_end(x_paths_end))

        x = torch.cat((x_velocity, x_center_path_distance, x_center_path, x_front, x_paths_start, x_paths_end), dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc6(x))
        z = self.fc5(x)
        z = torch.sigmoid(z)

        y = self.fc10(x)
        y = torch.tanh(y)
        return torch.cat((z, y), dim=1)

class ContinuousQAgent(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        self.is_q_function = True
        self.model = build_mlp(
            [state_dim + action_dim] + list(hidden_layers) + [1], activation=nn.ReLU()
        )

    def forward(self, t):
        # Get the current state $s_t$ and the chosen action $a_t$
        X_velocity = self.get(("env/env_obs/velocity", t))
        X_center_path_distance = self.get(("env/env_obs/center_path_distance", t))
        X_center_path = self.get(("env/env_obs/center_path", t))
        X_front = self.get(("env/env_obs/front", t))
        X_paths_start = self.get(("env/env_obs/paths_start", t)).permute(1, 0, 2)[0]
        X_paths_end = self.get(("env/env_obs/paths_end", t)).permute(1, 0, 2)[0]

        obs = torch.cat((X_velocity, X_center_path_distance, X_center_path, X_front, X_paths_start, X_paths_end), dim=1)  # shape B x D_{obs}

        action = self.get(("action", t))  # shape B x D_{action}

        # Compute the Q-value(s_t, a_t)
        obs_act = torch.cat((obs, action), dim=1)  # shape B x (D_{obs} + D_{action})
        # Get the q-value (and remove the last dimension since it is a scalar)
        q_value = self.model(obs_act).squeeze(-1)
        self.set((f"{self.prefix}q_value", t), q_value)

class ContinuousDeterministicActor(Agent):
    def __init__(self):
        super().__init__()
                
        model = Net_specifique(1,3,1,3,3,3,3,1)
        model.load_state_dict(torch.load('crab_model_pre_training.pth'))
        self.model = model

    def forward(self, t, **kwargs):
        X_max_steer_angle = self.get(("env/env_obs/max_steer_angle", t))
        X_velocity = self.get(("env/env_obs/velocity", t))
        X_center_path_distance = self.get(("env/env_obs/center_path_distance", t))
        X_center_path = self.get(("env/env_obs/center_path", t))
        X_front = self.get(("env/env_obs/front", t))
        X_paths_start = self.get(("env/env_obs/paths_start", t)).permute(1, 0, 2)[0]
        X_paths_end = self.get(("env/env_obs/paths_end", t)).permute(1, 0, 2)[0]
        X_paths_width = self.get(("env/env_obs/paths_width", t)).permute(1, 0, 2)[0]
        action = self.model(X_max_steer_angle, X_velocity, X_center_path_distance, X_center_path, X_front, X_paths_start, X_paths_end, X_paths_width)
        self.set(("action", t), action)
class AddGaussianNoise(Agent):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, t, **kwargs):
        act = self.get(("action", t))
        dist = Normal(act, self.sigma)
        action = dist.sample()
        self.set(("action", t), action)
class AddOUNoise(Agent):
    """
    Ornstein-Uhlenbeck process noise for actions as suggested by DDPG paper
    """

    def __init__(self, std_dev, theta=0.15, dt=1e-2):
        self.theta = theta
        self.std_dev = std_dev
        self.dt = dt
        self.x_prev = 0

    def forward(self, t, **kwargs):
        act = self.get(("action", t))
        x = (
            self.x_prev
            + self.theta * (act - self.x_prev) * self.dt
            + self.std_dev * math.sqrt(self.dt) * torch.randn(act.shape)
        )
        self.x_prev = x
        self.set(("action", t), x)