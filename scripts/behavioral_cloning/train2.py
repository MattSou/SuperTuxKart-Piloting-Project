# Train an MLP to comoute actions from observations from a dataset 
# produced by a "create_dataset.py" type of script
# Environment : discrete-v0 (1 discrete action between 0 and 1119)

import torch
import json
import argparse
from tqdm import tqdm
from bbrl_utils.nn import build_mlp
from torch.utils.data import DataLoader, TensorDataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='Dataset path')
parser.add_argument('--n_epochs', type=int, help='Number of epochs')
parser.add_argument('--suffix', type=str, help="suffix to add at the end of the .pth files")
args = parser.parse_args()

suffix = args.suffix

hidden_sizes = [300, 300, 300, 300]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(args.dataset, 'r') as f:
    dataset = json.load(f)

obs = torch.FloatTensor(dataset['obs'])
actions = torch.LongTensor(dataset['action'])

obs = obs.to(device)
actions = actions.to(device)

train_dataset = TensorDataset(obs, actions)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = build_mlp([obs.size(1)] + hidden_sizes + [5*2*2*2*2*2*7], activation=torch.nn.ReLU())

model.to(device)

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

torch.save(model, f'model_discretev0_{suffix}.pth')
