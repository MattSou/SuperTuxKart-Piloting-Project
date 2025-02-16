import torch
import json
from pystk2_gymnasium import AgentSpec
from functools import partial
import torch
from bbrl.agents.gymnasium import make_env
from pystk2_gymnasium.stk_wrappers import PolarObservations, ConstantSizedObservations, OnlyContinuousActionsWrapper, DiscreteActionsWrapper

import numpy as np
import numpy as np
from argparse import ArgumentParser

from gail_utils import get_data_by_learner_policy, get_data_by_expert, \
      get_discriminator_loss, get_policy_loss, STATE_SIZE, ACTION_SIZE, \
        PolicyNet, ValueNet, DiscriminatorNet
# from gridworld import GridWorld


parser = ArgumentParser()
parser.add_argument("--data_path", type=str, default='dataset_gail.json')
parser.add_argument("--num_samples", type=int, default=2000)
parser.add_argument("--max_timestep", type=int, default=200)
parser.add_argument("--gamma", type=float, default=0.995)
parser.add_argument("--gae_lambda", type=float, default=1.0)
parser.add_argument("--vf_coeff", type=float, default=0.01)
parser.add_argument("--kl_coeff", type=float, default=1.0)
parser.add_argument("--_lambda", type=float, default=0.005)
parser.add_argument("--num_iter", type=int, default=1000)
args = parser.parse_args()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_wrappers():
    """Returns a list of additional wrappers to be applied to the base
    environment"""
    return [
        # Example of a custom wrapper
        lambda env: PolarObservations(ConstantSizedObservations(env)),
        lambda env: OnlyContinuousActionsWrapper(env),
        lambda env: DiscreteActionsWrapper(env)
        #lambda env: MyWrapper(env),
    ]

make_stkenv = partial(
        make_env,
        "supertuxkart/full-v0",
        wrappers = get_wrappers(),
        render_mode=None,
        autoreset=True,
        agent=AgentSpec(use_ai=False, name='player_name'),
    )
env = make_stkenv()


# load expert data from pickle
with open(f"{args.data_path}", "rb") as f:
    exp_data = json.load(f)
#
# Generate model
#
policy_func = PolicyNet([512,512,512]).to(device)
policy_func.model.load_state_dict(torch.load('gail_pretrain.pth'))
value_func = ValueNet().to(device)
discriminator = DiscriminatorNet().to(device)

obs = exp_data['obs']
X_velocity = torch.tensor(obs["velocity"])
X_center_path_distance = torch.tensor(obs["center_path_distance"])
X_center_path = torch.tensor(obs["center_path"])
print(X_center_path[0:2])
X_front = torch.tensor(obs["front"])
print(X_front[0:2])
X_paths_start = torch.tensor(obs["paths_start"]).permute(1, 0, 2)[0]
print(X_paths_start[0:2])
X_paths_end = torch.tensor(obs["paths_end"]).permute(1, 0, 2)[0]
exp_states = torch.cat([X_velocity, X_center_path_distance, X_center_path, X_front, X_paths_start, X_paths_end], dim=-1)
print(exp_states[0:2])
# quit()

# obs = torch.cat((
#     torch.tensor(obs[:,-3:]),
#     torch.tensor(obs[:,13:14]),
#     torch.tensor(obs[:,10:13]),
#     torch.tensor(obs[:,7:10]),
#     torch.tensor(obs[:,16:19]),
#     torch.tensor(obs[:,75:78]),
#     torch.tensor(obs[:,51:54]),
# ))


action = np.array(exp_data['action'])
# nvec = [5,2,2,2,2,2,7]
# new_action = np.zeros((len(action), 7))
# for i in range(len(action)):
#     actions = []
#     for n in nvec:
#         actions.append(action[i] % n)
#         action[i] = action[i] // n
#     new_action[i] = np.array(actions)

new_action = action[:,1:2]
print(new_action[0:2])
# quit()


exp_states = exp_states.to(device)
exp_actions = torch.tensor(new_action, dtype=torch.float32).to(device)

from torch.utils.data import DataLoader
import itertools


expert_loader = DataLoader(
    list(zip(exp_states, exp_actions)),
    batch_size=args.num_samples,
    shuffle=False,
)
expert_iter = iter(itertools.cycle(expert_loader))



reward_records = []

opt_d = torch.optim.AdamW(discriminator.parameters(), lr=0.001)
opt_pi = torch.optim.AdamW(list(policy_func.parameters()) + list(value_func.parameters()), lr=0.001)

env = make_stkenv()

for iter_num in range(args.num_iter):
    # get expert data
    states_ex, actions_ex = get_data_by_expert(expert_iter, args.num_samples)
    

    # get data by policy pi
    states, actions, logits, advantages, discount, reward_mean = get_data_by_learner_policy(
        env=env,
        policy_net=policy_func,
        value_net=value_func,
        discriminator_net=discriminator,
        num_samples=600, 
        max_timestep=args.max_timestep,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
    )
    reward_records.append(reward_mean.item())

    # update discriminator
    d_loss = get_discriminator_loss(
        discriminator,
        states_ex,
        actions_ex,
        states,
        actions,
    )
    opt_d.zero_grad()
    d_loss.backward()
    opt_d.step()

    # update policy
    adv_loss, val_loss, kl_loss, ent_loss = get_policy_loss(
        policy_func,
        states,
        actions,
        logits,
        advantages,
        discount,
    )
    pi_loss = adv_loss + val_loss * args.vf_coeff + kl_loss *args.kl_coeff + ent_loss * args._lambda
    opt_pi.zero_grad()
    pi_loss.mean().backward()
    opt_pi.step()


    # output log
    if iter_num % 20 == 0:
        line_end = "\n"
        print_reward = np.average(reward_records[-200:])
    else:
        line_end = "\r"
        print_reward = reward_records[-1]
        
    print("iter{} - reward mean {:2.4f} - d_loss {:2.4f} - pi_loss {:2.4f}".format( iter_num, print_reward, d_loss.item(), pi_loss.mean().item()), end=line_end)
    
    torch.save(policy_func.model.state_dict(), 'gail_policy.pth')
    # stop if reward mean reaches to threshold
    # if np.average(reward_records[-20:]) > 5.0:
    #     break

print("\nDone")
