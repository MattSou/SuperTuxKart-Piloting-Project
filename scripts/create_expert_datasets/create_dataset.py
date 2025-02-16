# Dataset creation from a driectory of logs
# in the flattened-v0 environment

from os import path
import os
import subprocess
import argparse
from tqdm import tqdm
import json

parser = argparse.ArgumentParser()

parser.add_argument('--log_dir', type=str, help='Log directory')
parser.add_argument('--output_path', type=str, help='Output dataset path')

args = parser.parse_args()

log_dir = args.log_dir

assert path.exists(log_dir), f'Invalid log directory {log_dir}'

list_existing_logs = os.listdir(log_dir)

assert len(list_existing_logs) > 0, f'No logs found in {log_dir}'

dataset = {'obs':[], 'discrete_actions':[], 'continuous_actions':[]}

for log in list_existing_logs:
    assert log.endswith('.json'), f'Invalid log extension, should end with .json but found {log}'

    log_path = path.join(log_dir, log)
    with open(log_path, 'r') as f:
        data = json.load(f)
    
    assert 'overall_observations' in data.keys()

    data = data['overall_observations']

    for i in range(len(data)):
        assert list(data[i]['0'].keys()) == ['discrete', 'continuous', 'action'], f'Invalid keys found in {log}'
        dataset['obs'].append(data[i]['0']['discrete'] + data[i]['0']['continuous'])
        dataset['discrete_actions'].append(data[i]['0']['action']['discrete'])
        dataset['continuous_actions'].append(data[i]['0']['action']['continuous'])

output_path = args.output_path

with open(output_path, 'w') as f:
    json.dump(dataset, f)
    

