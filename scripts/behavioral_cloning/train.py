# Train an MLP to comoute actions from observations from a dataset 
# produced by a "create_dataset.py" type of script
# Environment : flattened-v0 (2 continuous actions and 5 discrete actions)

import torch
import json
import argparse
from tqdm import tqdm

from torch.utils.data import DataLoader, TensorDataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='Dataset path')
parser.add_argument('--n_epochs', type=int, help='Number of epochs')
parser.add_argument('--suffix', type=str, help="suffix to add at the end of the .pth files")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

suffix = args.suffix

with open(args.dataset, 'r') as f:
    dataset = json.load(f)

# print(len(dataset['obs']))
# quit()

obs = torch.FloatTensor(dataset['obs'])
discrete_actions = torch.FloatTensor(dataset['discrete_actions'])
continuous_actions = torch.FloatTensor(dataset['continuous_actions'])

obs = obs.to(device)
discrete_actions = discrete_actions.to(device)
continuous_actions = continuous_actions.to(device)

train_dataset = TensorDataset(obs, discrete_actions, continuous_actions)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model_discrete = torch.nn.Sequential(
    torch.nn.Linear(obs.size(1), 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, discrete_actions.size(1)),
    torch.nn.Sigmoid()
)

model_continuous = torch.nn.Sequential(
    torch.nn.Linear(obs.size(1), 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, continuous_actions.size(1))
)

model_continuous.to(device)
model_discrete.to(device)

discrete_criterion = torch.nn.BCELoss()
continuous_criterion = torch.nn.MSELoss()

optimizer = torch.optim.Adam(list(model_discrete.parameters()) + list(model_continuous.parameters()), lr=1e-3)
for i in range(args.n_epochs):
    total_loss1, total_loss2,  = 0, 0
    for obs, discrete_actions, continuous_actions in tqdm(train_loader):
        optimizer.zero_grad()
        discrete_pred = model_discrete(obs)
        continuous_pred = model_continuous(obs)
        # print(discrete_pred, continuous_pred)
        # print(discrete_pred.shape, continuous_pred.shape)
        # print(discrete_actions, continuous_actions)
        # print(discrete_actions.shape, continuous_actions.shape)
        loss1 = discrete_criterion(discrete_pred, discrete_actions) 
        loss2 = continuous_criterion(continuous_pred, continuous_actions)
        loss = loss1 + loss2
        total_loss1 += loss1.item()
        total_loss2 += loss2.item()
        loss.backward()
        optimizer.step()
    total_loss = total_loss1 + total_loss2
    print(f'Epoch {i}, Loss1 {total_loss1/len(train_loader)}, Loss2 {total_loss2/len(train_loader)}, Total Loss {total_loss/len(train_loader)}')

torch.save(model_discrete, f'model_discrete_{suffix}.pth')
torch.save(model_continuous, f'model_continuous_{suffix}.pth')