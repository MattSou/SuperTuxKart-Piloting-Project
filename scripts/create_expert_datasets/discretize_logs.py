# Produces a directory of discrte logs equivalent 
# to the flattened logs in the source directory

import argparse
import json
import os
from pystk2_gymnasium.utils import Discretizer
from gymnasium import spaces
import copy

parser = argparse.ArgumentParser()

parser.add_argument('--logs_dir', type=str, help='Logs directory')
parser.add_argument('--output_dir', type=str, help='Output dataset path')

args = parser.parse_args()


logs_dir = args.logs_dir
output_dir = args.output_dir

assert os.path.exists(logs_dir), f'Invalid logs directory {logs_dir}'

list_existing_logs = os.listdir(logs_dir)

assert len(list_existing_logs) > 0, f'No logs found in {logs_dir}'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

d_acceleretaion = Discretizer(spaces.Box(0, 1, shape=(1,)), 5)
d_steer = Discretizer(spaces.Box(-1, 1, shape=(1,)), 7)
nvec = [5,2,2,2,2,2,7]


for log in list_existing_logs:
    with open(os.path.join(logs_dir, log), 'r') as f:
        data = json.load(f)
    discretized_log = copy.deepcopy(data)
    overall_observations = []
    N = len(discretized_log["overall_observations"])
    for i in range(N):
        # print(i)
        action = data["overall_observations"][i]["0"]["action"]
        actions = [d_acceleretaion.discretize(action["continuous"][0])] +[int(x) for x in action["discrete"]]+ [d_steer.discretize(action["continuous"][1])]
        
        ## AT THIS POINT, ACTIONS ARE DISCRETIZED FOR multidiscrete-v0 environment
        ## CUT HERE IF YOU WANT LOGS FOR THIS ENVIRONMENT
        
        # print(actions)
        act = actions[-1]
        for i in range(len(nvec)-2, 0, -1):
            # print(act)
            n = nvec[i]
            act = act*n
            act += actions[i-1]        
        
        ## AT THIS POINT, ACTIONS ARE DISCRETIZED FOR discrete-v0 environment

        # print(act)
        overall_observations.append({
            '0':{
                'discrete': data["overall_observations"][i]["0"]["discrete"],
                'continuous': data["overall_observations"][i]["0"]["continuous"],
                'action': act
            }
        })
    # print(overall_observations[0]["0"]["action"])
    discretized_log["overall_observations"] = overall_observations
    # print(discretized_log["overall_observations"][0]["0"]["action"])

    with open(os.path.join(output_dir, log.replace("flattened_v0", "discrete_v0")), 'w') as f:
        json.dump(discretized_log, f)
