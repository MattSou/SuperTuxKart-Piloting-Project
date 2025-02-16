import torch
import json
from pystk2_gymnasium import AgentSpec
from functools import partial
import torch
import inspect
from bbrl.agents.gymnasium import ParallelGymAgent, make_env
import numpy as np
import gymnasium as gym
# from gridworld import GridWorld

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.nn as nn
from torch.nn import functional as F
from bbrl_utils.nn import build_mlp

STATE_SIZE, ACTION_SIZE = 16, 7

# class PolicyNet(nn.Module):
#     def __init__(self, hidden_dim=64):
#         super().__init__()
#         self.hidden = nn.Linear(STATE_SIZE, hidden_dim)
#         self.output = nn.Linear(hidden_dim, ACTION_SIZE)

#     def forward(self, s):
#         outs = self.hidden(s)
#         outs = F.relu(outs)
#         logits = self.output(outs)
#         return logits

class PolicyNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.model = build_mlp([STATE_SIZE] + list(hidden_dim) +[ACTION_SIZE], activation=nn.ReLU())
        print(self.model)

    def forward(self, s):
        # X_velocity = s["velocity"]
        # X_center_path_distance = s["center_path_distance"]
        # X_center_path = s["center_path"]
        # X_front = s["front"]
        # X_paths_start = s["paths_start"].permute(1, 0, 2)[0]
        # X_paths_end = s["paths_end"].permute(1, 0, 2)[0]
        # observation = torch.cat([X_velocity, X_center_path_distance, X_center_path, X_front, X_paths_start, X_paths_end], dim=-1)
        logits = self.model(s)
        return logits

class ValueNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden = nn.Linear(STATE_SIZE, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, s):
        outs = self.hidden(s)
        outs = F.relu(outs)
        value = self.output(outs)
        return value

class DiscriminatorNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden1 = nn.Linear(STATE_SIZE + ACTION_SIZE, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.get_logits = nn.Linear(hidden_dim, 1)
        self.get_sigmoid = nn.Sigmoid()

    def forward(self, s):
        outs = self.hidden1(s)
        outs = F.relu(outs)
        outs = self.hidden2(outs)
        outs = F.relu(outs)
        logits = self.get_logits(outs)
        output = self.get_sigmoid(logits)
        return output

#
# Generate model
#
from torch.utils.data import DataLoader
import itertools

def get_data_by_expert(expert_iter, num_samples):
    data_length = 0
    while data_length < num_samples:
        states, actions = next(expert_iter)
        data_length = len(states)
    return states, actions.long()


def get_data_by_learner_policy(
    env,
    policy_net,
    value_net,
    discriminator_net,
    gamma,
    gae_lambda,
    num_samples,
    batch_size=1,
    max_timestep=200
):
    """
    Collect samples with policy pi.
    To speed up training, this function runs as a batch.

    Parameters
    ----------
    env : GridWorld
        Environment class.
    policy_net : torch.nn.Module
        Policy network to pick up action.
    value_net : torch.nn.Module
        Value network used to get values and advantages.
    discriminator_net : torch.nn.Module
        Discriminator network used to get values and advantages
    gamma : float
        A discount value.
    gae_lambda : float
        A parameter controlling bias and variance in GAE. (See above)
    num_samples : int
        Number of samples to pick up.
    batch_size : int
        Batch size used to pick up samples.

    Returns
    ----------
    states : torch.tensor((num_samples), dtype=int)
        Collected states.
    actions : torch.tensor((num_samples), dtype=int)
        Collected actions.
    action_logits : torch.tensor((num_samples, ACTION_SIZE), dtype=float)
        Logits used to pick up actions.
    advantages : torch.tensor((num_samples), dtype=float)
        Advantages which is used to optimize policy.
        This advantage is obtained by GAE (generalized advantage estimation).
        This tensor has graph to be optimized (i.e, can be used for optimization.)
    discount : torch.tensor((num_samples), dtype=float)
        Discount factor gamma^t.
        Later this coefficient is used to get gamma-discounted causal entropy.
    average_reward : torch.tensor(float)
        The average of episode's reward in all executed episodes.
        This reward is not used in GAIL algorithm,
        but it's used for the evaluation of training in this example.
    """

    ##########
    # Operations are processed as a batch.
    # All working tensor has dimension: (step_count, batch_size, ...)
    ##########

    # initialize results
    states = torch.empty((0, STATE_SIZE), dtype=float).to(device)
    actions = torch.empty((0), dtype=int).to(device)
    action_logits = torch.empty((0, ACTION_SIZE), dtype=float).to(device)
    advantages = torch.empty((0), dtype=float).to(device)
    discount = torch.empty((0), dtype=float).to(device)

    # note : reward is not used in GAIL, but it's used to evaluate how well it's learned...
    episode_rewards = torch.empty((0)).to(device)
    step_count =0
    s = torch.empty((0)).to(device)
    while len(states) < num_samples:
        
        # initialize episode
        # print("iter{}".format(step_count), end="\r")
        #
        if s.nelement() == 0:
            reward_total = torch.zeros(batch_size).to(device)
            s, _ = env.reset()
            X_velocity = torch.tensor(s["velocity"])
            X_center_path_distance = torch.tensor(s["center_path_distance"])
            X_center_path = torch.tensor(s["center_path"])
            # print(X_center_path[0:2])
            X_front = torch.tensor(s["front"])
            # print(X_front[0:2])
            X_paths_start = torch.tensor(s["paths_start"])[0]
            # print(X_paths_start.shape)

            # print(X_paths_start[0:2])
            X_paths_end = torch.tensor(s["paths_end"])[0]
            s = torch.cat([X_velocity, X_center_path_distance, X_center_path, X_front, X_paths_start, X_paths_end], dim=-1).to(device)
            # s = torch.cat([torch.tensor(s['discrete'], dtype=torch.float32), torch.tensor(s['continuous'], dtype=torch.float32)]).to(device)
            states_ep = []
            actions_ep = []
            action_logits_ep = []

        #
        # step episode
        #

        # get state
        states_ep.append(s.unsqueeze(dim=0))
        # step_count = states_ep.shape[1]
        # print("iter{}".format(step_count), end="\r")
        # get action with policy pi
        # s_onehot = F.one_hot(s, num_classes=STATE_SIZE)
        logits = policy_net(s).detach()
        # print(logits)
        probs = F.softmax(logits, dim=-1)
        a = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)
        action_for_step = {'acceleration': 4, 'steer': a}
        # print(a)
        actions_ep.append(a.unsqueeze(dim=0))
        action_logits_ep.append(logits.unsqueeze(dim=0))

        # step to the next state
        s, r, done, _, _ = env.step(action_for_step)
        step_count += 1
        X_velocity = torch.tensor(s["velocity"])
        X_center_path_distance = torch.tensor(s["center_path_distance"])
        X_center_path = torch.tensor(s["center_path"])
        # print(X_center_path[0:2])
        X_front = torch.tensor(s["front"])
        # print(X_front[0:2])
        X_paths_start = torch.tensor(s["paths_start"])[0]
        # print(X_paths_start[0:2])
        X_paths_end = torch.tensor(s["paths_end"])[0]
        s = torch.cat([X_velocity, X_center_path_distance, X_center_path, X_front, X_paths_start, X_paths_end], dim=-1).to(device)
        # s = torch.cat([torch.tensor(s['discrete'], dtype=torch.float32), torch.tensor(s['continuous'], dtype=torch.float32)]).to(device)
        # (note : reward is only used to evaluate)
        reward_total += r
        done = torch.tensor(done).to(device)
        # print(s.shape)  
        # print(states_ep.shape)
        #
        # finalize episode
        #

        # pick up indices to be done (not truncated) and add to reward's record
        # done_indices = done.nonzero().squeeze(dim=-1)
        if step_count==max_timestep:
            episode_rewards = torch.cat((episode_rewards, reward_total))
        elif done:
            episode_rewards = torch.cat((episode_rewards, reward_total))
        # pick up indices to be finalized (terminated or truncated)
        trunc = torch.tensor((step_count==max_timestep) or (step_count >= num_samples - len(states))).to(device)
        fin = torch.logical_or(done, trunc)
        # fin_indices = fin.nonzero().squeeze(dim=-1)
        if fin:
            states_ep = torch.stack(states_ep).squeeze(1)
            actions_ep = torch.stack(actions_ep)
            action_logits_ep = torch.stack(action_logits_ep)
            # fin_len = len(fin_indices)
            # pick up results to be finalized
            
            # get log(D(s,a))
            # print(states_ep.shape)
            # print(actions_ep.shape)
            actions_ep_onehot = F.one_hot(actions_ep, num_classes=ACTION_SIZE).float().squeeze(dim=1)
            # print(actions_ep_onehot.shape)
            state_action_fin = torch.cat((states_ep, actions_ep_onehot), dim=1)
            # print(state_action_fin.shape)
            d_log_fin = torch.log(discriminator_net(state_action_fin).detach().squeeze(dim=-1)) # detach() - gradient update in discriminator is not required
            # get values and value loss (see above for TD)
            values_current_fin = value_net(states_ep).squeeze(dim=-1)
            # when it's truncated, set next value in last element. 0 otherwise.
            if trunc:
                state_last = s
                value_last = value_net(s).squeeze(dim=-1)
            else:
                value_last = torch.zeros(1).to(device)
            values_next_fin = torch.cat((values_current_fin[1:], value_last.unsqueeze(dim=0)), dim=0)
            # get delta
            delta_fin = d_log_fin + values_next_fin * gamma - values_current_fin
            # print(delta_fin.shape)
            # get advantages (see above for GAE)
            gae_params = torch.tensor([(gamma * gae_lambda)**i for i in range(len(delta_fin))]).to(device)
            advs_fin = [torch.sum(gae_params[:len(delta_fin)-i] * delta_fin[i:]) for i in range(len(delta_fin))]
            advs_fin = torch.stack(advs_fin)
            # print(advs_fin.shape)
            # get gamma-discount
            discount_fin = torch.tensor([gamma**i for i in range(len(delta_fin))]).to(device)
            # print(discount_fin.shape)
            # add to results
            states = torch.cat((states, states_ep))
            actions = torch.cat((actions, actions_ep.flatten()))
            action_logits = torch.cat((action_logits, action_logits_ep.reshape(-1, ACTION_SIZE)))
            advantages = torch.cat((advantages, advs_fin))
            discount = torch.cat((discount, discount_fin))
            # remove finalized items in batch
            s = torch.empty((0)).to(device)
            step_count = 0
            

    # truncate results
    states = states[:num_samples]
    actions = actions[:num_samples]
    action_logits = action_logits[:num_samples,:]
    advantages = advantages[:num_samples]
    discount = discount[:num_samples]
    # shuffle results
    rnd_indices = torch.randperm(num_samples)
    states = states[rnd_indices]
    actions = actions[rnd_indices]
    action_logits = action_logits[rnd_indices,:]
    advantages = advantages[rnd_indices]
    discount = discount[rnd_indices]

    return states, actions, action_logits, advantages, discount, torch.mean(episode_rewards)

def get_discriminator_loss(discriminator_net, exp_states, exp_actions, pi_states, pi_actions):
    """
    Collect samples with policy pi.
    To speed up training, this function runs as batch.

    Parameters
    ----------
    discriminator_net : torch.nn.Module
        Discriminator network to be updated
    exp_states : torch.tensor((num_samples), dtype=int)
        States visited by expert policy.
    exp_actions : torch.tensor((num_samples), dtype=int)
        Corresponding actions to be taken by expert.
    pi_states : torch.tensor((num_samples), dtype=int)
        States visited by policy pi.
    pi_actions : torch.tensor((num_samples), dtype=int)
        Corresponding actions to be taken by policy pi.

    Returns
    ----------
    Mean of discriminator loss
    """

    # get D(s,a)
    actions_onehot_pi = F.one_hot(pi_actions, num_classes=ACTION_SIZE).float()
    # print(pi_states.shape, actions_onehot_pi.shape)
    state_action_pi = torch.cat((pi_states, actions_onehot_pi), dim=-1).float()
    d_pi = discriminator_net(state_action_pi).squeeze(dim=-1)


    actions_onehot_exp = F.one_hot(exp_actions, num_classes=ACTION_SIZE).float().squeeze(1)
    # print(exp_states.shape, actions_onehot_exp.shape)
    state_action_exp = torch.cat((exp_states, actions_onehot_exp), dim=-1)
    d_exp = discriminator_net(state_action_exp).squeeze(dim=-1)

    # get mean of binary cross entropy (BCE) loss
    mean_loss_pi = F.binary_cross_entropy(d_pi, torch.ones_like(d_pi).to(device))
    mean_loss_exp = F.binary_cross_entropy(d_exp, torch.zeros_like(d_exp).to(device))

    return mean_loss_pi + mean_loss_exp

def get_policy_loss(policy_net, states, actions, logits, advantages, discount):
    logits_old = logits

    # get logits to be used for optimization
    
    logits_new = policy_net(states.float())

    # get advantage loss (see above)
    logprb_old = -F.cross_entropy(logits_old, actions, reduction="none") # get log probability (see above note)
    logprb_new = -F.cross_entropy(logits_new, actions, reduction="none") # get log probability (see above note)
    prb_ratio = torch.exp(logprb_new - logprb_old) # P(a|pi_new(s)) / P(a|pi_old(s))
    advantage_loss = prb_ratio * advantages

    # get value loss (see above)
    value_loss = torch.mean(advantages**2)

    # get KL loss
    # (see https://github.com/tsmatz/reinforcement-learning-tutorials/blob/master/04-ppo.ipynb)
    l_old = logits_old - torch.amax(logits_old, dim=1, keepdim=True) # reduce quantity
    l_new = logits_new - torch.amax(logits_new, dim=1, keepdim=True) # reduce quantity
    e_old = torch.exp(l_old)
    e_new = torch.exp(l_new)
    e_sum_old = torch.sum(e_old, dim=1, keepdim=True)
    e_sum_new = torch.sum(e_new, dim=1, keepdim=True)
    p_old = e_old / e_sum_old
    kl_loss = torch.sum(
        p_old * (l_old - torch.log(e_sum_old) - l_new + torch.log(e_sum_new)),
        dim=1,
        keepdim=True)

    # get gamma-discounted causal entropy loss (see above)
    entropy_loss = -discount * logprb_new

    return advantage_loss, value_loss, kl_loss, entropy_loss
import numpy as np


