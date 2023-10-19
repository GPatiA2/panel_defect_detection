from models.torchTraditionalModel import TorchTraditionalClassifier
import os
import cv2
from torch.utils.data import DataLoader
import json
import torch
import sklearn.model_selection as sk
import torch.nn as nn
from graphviz import Digraph
import torch
from torchviz import make_dot

epochs = 500

model = TorchTraditionalClassifier(35,50)
state_dict = model.state_dict()
initial_params = model.get_initial_params()
initial_weights = torch.FloatTensor(initial_params)
state_dict['weights'] = initial_weights
model.load_state_dict(state_dict)
transforms = model.transforms()

print(list(model.parameters()))
input()

with open('dataset/blob_dataset.json', 'r') as f:
    data = json.load(f)

x = []
y = []

training_frac = .8

for k, v in data['train'].items():
    im = cv2.imread(os.path.join('dataset/all', k))
    im = transforms(im)
    x.append(im)
    y.append(v)

X_train, X_val, y_train, y_val = sk.train_test_split(x, y, test_size= 1 - training_frac, shuffle=True, stratify=y)

train_dataset = list(zip(X_train, y_train))
val_dataset   = list(zip(X_val, y_val))

train_dl = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
val_dl   = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)

criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

for i in range(len(model.weights)):
    print(model.weights[i].data)

input()

batch = 0
for i in range(epochs):
    for x, y in train_dl:
        optimizer.zero_grad()
        y_pred = model(x)
        make_dot(y_pred).view()
        input()
        y_pred = y_pred.type(torch.float32)
        y = y.type(torch.float32)
        loss = criterion(y_pred, y)
        loss.requires_grad = True
        loss.backward()
        optimizer.step()
        batch += 1

    print("Epoch: ", i, " Loss: ", loss.item())

print("MODEL PARAMS: ")
for i in range(len(model.weights)):
    print(model.weights[i].data)

