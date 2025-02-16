# Try of prtraining an agent in the discrete-v0 environment
# to start with an agent predicting more often the most frequent
# actions in the dataset

import torch
import json
import argparse
from tqdm import tqdm
from bbrl_utils.nn import build_mlp
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='Dataset path')
parser.add_argument('--n_epochs', type=int, help='Number of epochs')
args = parser.parse_args()

hidden_sizes = [300, 300, 300, 300]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(args.dataset, 'r') as f:
    dataset = json.load(f)

actions = dataset['action']
unique_actions, counts = np.unique(actions, return_counts=True)
# print(unique_actions, counts)
idx = np.argsort(counts)
counts = counts[idx]
actions_to_keep = unique_actions[idx]

idx = np.where(counts > 2000)[0]
actions_to_keep = actions_to_keep[idx]
counts = counts[idx]
# print(actions_to_keep, counts)

probs = counts/len(actions)
probs_other = (1 - np.sum(probs))/(5*2*2*2*2*2*7 - len(actions_to_keep))
print(probs,probs_other)

probs = np.array([probs[np.where(actions_to_keep==i)][0] if i in actions_to_keep else probs_other for i in range(5*2*2*2*2*2*7)])

print(probs.shape, np.sum(probs))
probs = probs/np.sum(probs)
print(probs.shape)

mu, sigma = np.mean(np.array(dataset['obs'])[:, 8:], axis=0), np.std(np.array(dataset['obs'])[:, 8:], axis=0)
print(mu.shape, sigma.shape)
print(np.array(dataset['obs'])[:,:8])
occurences, count_occurences = np.unique(np.array(dataset['obs'])[:,:8], axis=0, return_counts=True)
print(occurences)
prob_occurences = count_occurences/np.sum(count_occurences)

N = 100000
p = np.array(dataset['obs']).shape[1]
obs = np.zeros((N, p))
actions = np.zeros(N)
for i in range(len(obs)):
    obs[i, :8] = occurences[np.random.choice(len(occurences), p=prob_occurences)]
    obs[i, 8:] = np.random.normal(mu, sigma)
    actions[i] = np.random.choice(5*2*2*2*2*2*7, p=probs)
print(np.unique(actions, return_counts=True))

print(obs.shape, actions.shape)
obs = torch.FloatTensor(obs)
actions = torch.LongTensor(actions)


obs = obs.to(device)
actions = actions.to(device)

train_dataset = TensorDataset(obs, actions)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = build_mlp([obs.size(1)] + hidden_sizes + [5*2*2*2*2*2*7], activation=torch.nn.ReLU())

model.to(device)

# weight = np.array([probs[np.where(actions_to_keep==i)][0] if i in actions_to_keep else probs_other for i in range(5*2*2*2*2*2*7)])
# # print(weight)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for i in range(args.n_epochs):
    total_loss = 0
    for obs, actions in tqdm(train_loader):
        optimizer.zero_grad()
        pred = model(obs)
        # print(pred.shape, actions.shape)
        # print(max(actions), min(actions))
        # print(discrete_pred, continuous_pred)
        # print(discrete_pred.shape, continuous_pred.shape)
        # print(discrete_actions, continuous_actions)
        # print(discrete_actions.shape, continuous_actions.shape)
        loss = criterion(pred, actions)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {i}, Loss {total_loss/len(train_loader)}')

torch.save(model.state_dict(), 'model_discrete_true.pth')

obs = np.zeros((N, p))
model.eval()
actions = np.zeros(N)
with torch.no_grad():
    for i in range(len(obs)):
        obs[i, :8] = occurences[np.random.choice(len(occurences), p=prob_occurences)]
        obs[i, 8:] = np.random.normal(mu, sigma)
        actions[i] = torch.argmax(model(torch.FloatTensor(obs[i]).to(device)))

print(np.unique(actions, return_counts=True))
