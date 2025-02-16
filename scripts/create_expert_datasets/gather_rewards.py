# Gathering final rewards from  the different trajectories
# stored in a lod directory

from os import path
import os
import subprocess
import argparse
from tqdm import tqdm
import json
parser = argparse.ArgumentParser()

parser.add_argument('--log_dir', type=str, help='Output log directory')

args = parser.parse_args()

dir = args.log_dir

TRACKS = [
    'abyss', 'black_forest', 'candela_city', 'cocoa_temple', 
    'cornfield_crossing', 'fortmagma', 'gran_paradiso_island', 
    'hacienda', 'lighthouse', 'mines', 'minigolf', 'olivermath', 
    'ravenbridge_mansion', 'sandtrack', 'scotland', 'snowmountain', 
    'snowtuxpeak', 'stk_enterprise', 'volcano_island',
    'xr591', 'zengarden'
    ]

list_existing_logs = os.listdir(dir)
rewards = {track: [] for track in TRACKS}

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

for log in list_existing_logs:
    log_path = path.join(dir, log)
    with open(log_path, 'r') as f:
        data = json.load(f)
    

    track = data['track']
    reward = data['results'][0]['reward']

    rewards[track].append(reward)

# for i in tqdm(range(n_races)):
#     log_path = path.join(dir, f'log_{prefix}{n_logs + i}.json')
#     subprocess.run(['master-dac', 'rld', 'stk-race', agent, '--hide', '--output', log_path, '--use_ai'])


with open('rewards.json', 'w') as f:
    json.dump(rewards, f)
print(f'Rewards saved in rewards.json')

#python gather_logs.py --agent player_flattened_v0.zip --n_races 2 --ouput_log_dir test_logs

