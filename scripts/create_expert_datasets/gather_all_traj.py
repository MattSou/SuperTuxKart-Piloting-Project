# Script for gathering trajectories from an AI bot
# You can choose the difficulty level of the bot (0 to 3),
# if you want the rewards (--get_reward option)
# It iterates over all tracks, do '--n_races' races per track and keep only unique trajectories (no duplicates)
# NB: The agent defines the envrionment in which trajectories are computed !

from os import path
import os
import subprocess
import argparse
from tqdm import tqdm
import json
parser = argparse.ArgumentParser()

parser.add_argument('--agent', type=str, help='Agent name')
parser.add_argument('--n_races', type=int, help='Number of races per track')
parser.add_argument('--output_log_dir', type=str, help='Output log directory')
parser.add_argument('--prefix', type=str, help='Prefix for log files', default='')

args = parser.parse_args()

agent = args.agent
dir = args.output_log_dir
n_races = args.n_races
prefix = args.prefix

TRACKS = [
    'abyss', 
    # 'black_forest', 
    'candela_city', 'cocoa_temple', 
    'cornfield_crossing', 'fortmagma', 'gran_paradiso_island', 
    'hacienda', 'lighthouse', 'mines', 'minigolf', 'olivermath', 
    'ravenbridge_mansion', 'sandtrack', 'scotland', 'snowmountain', 
    'snowtuxpeak', 'stk_enterprise', 'volcano_island',
    'xr591', 'zengarden'
    ]

if prefix != '':
    prefix = prefix + '_'

if not path.exists(dir):
    os.makedirs(dir)

list_existing_logs = os.listdir(dir)

# if len(list_existing_logs) == 0:
#     n_logs = 0
# else:
#     for log in list_existing_logs:
#         assert log.startswith('log_'), f'Invalid log name, should start with log_ but found {log}'
#         assert log.endswith('.json'), f'Invalid log extension, should end with .json but found {log}'
#         assert log[4:].split('.')[0].split('_')[-1].isdigit(), f'Invalid log name, should have a number after log_ but found {log}'

#     numbers = [int(log[4:].split('.')[0].split('_')[-1]) for log in list_existing_logs]
#     assert len(numbers) == len(set(numbers)), 'Duplicate log numbers found'
#     assert max(numbers) == len(numbers) - 1, f'missing log files, expected {max(numbers) + 1} logs but found {len(numbers)}'

#     n_logs = len(numbers)

for track in TRACKS:
    print(track)
    rewards = []
    n_logs = 0
    for i in tqdm(range(n_races)):
        log_path = path.join(dir, f'log_{prefix}{track}_{n_logs}.json')
        subprocess.run(['master-dac', 'rld', 'stk-race', agent, 
                        '--hide', '--output', log_path, '--use_ai', '--track', track, '--difficulty', '0', '--get_reward'])
        with open(log_path, 'r') as f:
            data = json.load(f)
        reward = data['results'][0]['reward']
        if reward not in rewards:
            rewards.append(reward)
            n_logs += 1
        print(rewards)
    if rewards.index(reward) != len(rewards) - 1 or len(rewards) ==1:
        os.remove(log_path)

# for i in tqdm(range(n_races)):
#     log_path = path.join(dir, f'log_{prefix}{n_logs + i}.json')
#     subprocess.run(['master-dac', 'rld', 'stk-race', agent, '--hide', '--output', log_path, '--use_ai'])

print(f'Logs saved in {dir}')

#python gather_logs.py --agent player_flattened_v0.zip --n_races 2 --ouput_log_dir test_logs

