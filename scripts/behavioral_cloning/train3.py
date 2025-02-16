# Train an MLP to comoute actions from observations from a dataset 
# produced by a "create_dataset.py" type of script
# Environment : multidiscrete-v0 (7 discrete actions)

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

hidden_sizes = [256, 256, 256, 256, 256]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(args.dataset, 'r') as f:
    dataset = json.load(f)

nvec = [5, 2, 2, 2, 2, 2, 7]

obs = torch.FloatTensor(dataset['obs'])
actions = torch.LongTensor(dataset['action'])

obs = obs.to(device)
actions = actions.to(device)

# def list_action(action):
#     actions = []
#     for n in nvec:
#         actions.append(action % n)
#         action = action // n
#     return actions

train_dataset = TensorDataset(obs, actions)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = build_mlp([obs.size(1)] + hidden_sizes + [5+2+2+2+2+2+7], activation=torch.nn.ReLU())

model.to(device)

criterion = torch.nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for i in range(args.n_epochs):
    total_loss = 0
    for obs, actions in tqdm(train_loader):
        optimizer.zero_grad()
        pred = model(obs)
        # actions = torch.stack([torch.LongTensor(list_action(a)) for a in action]).to(device)
        loss_0 = criterion(pred[:,:5], actions[:,0])
        loss_1 = criterion(pred[:,5:7], actions[:,1])
        loss_2 = criterion(pred[:,7:9], actions[:,2])
        loss_3 = criterion(pred[:,9:11], actions[:,3])
        loss_4 = criterion(pred[:,11:13], actions[:,4])
        loss_5 = criterion(pred[:,13:15], actions[:,5])
        loss_6 = criterion(pred[:,15:], actions[:,6])
        loss = 3*loss_0 + loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + 3*loss_6
        # print(pred.shape, actions.shape)
        # print(max(actions), min(actions))
        # print(discrete_pred, continuous_pred)
        # print(discrete_pred.shape, continuous_pred.shape)
        # print(discrete_actions, continuous_actions)
        # print(discrete_actions.shape, continuous_actions.shape)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {i}, Loss {total_loss/len(train_loader)}')

torch.save(model, f'model_multidiscrete_{suffix}.pth')
