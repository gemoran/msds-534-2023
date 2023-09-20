
## 
## Implementing a logistic regression model in PyTorch
## 
## We build logistic regression as 1-layer neural network
## 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader


n = 400
p = 2
x = np.random.uniform(-2, 2, size=(n, p))

beta0 = np.array([1, 1])

probs = np.exp(x @ beta0) / (1 + np.exp(x @ beta0))
y = np.random.binomial(1, probs)

#y = ((x[:,0] < 0) & (x[:, 1] > 0)).astype(x.dtype) + ((x[:,0] > 0) & (x[:, 1] < 0)).astype(x.dtype)

x = torch.tensor(x, dtype=torch.float)
y = torch.tensor(y, dtype=torch.float).view(-1, 1)

ntrain = 200
x_train = x[:ntrain]
y_train = y[:ntrain]
x_test = x[ntrain:]
y_test = y[ntrain:]

INPUT_DIM = x.shape[1]
OUT_DIM = 1

class LRModel(nn.Module):

    def __init__(self):
        super(LRModel, self).__init__()
        self.layer = nn.Linear(INPUT_DIM, OUT_DIM)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = self.layer(x)
        x = self.sigm(x)
        return(x)
    
model = LRModel()
optimizer = optim.SGD(model.parameters(), lr=0.5)

pred = model(x_train)

def loss_fn(y_pred, y_true):
    loss_vec = y_true * torch.log(y_pred + 1e-8) + (1 - y_true) * torch.log(1 - y_pred + 1e-8)
    output = -1.0 * torch.mean(loss_vec)
    return(output)

print("initial weights: " + str(model.layer.weight) + "\n")

num_iter = 200

for i in range(num_iter):
    print("iter :" + str(i))

    ## loss = nn.BCELoss()(model(x_train), y_train)
    loss = loss_fn(model(x_train), y_train.view(-1, 1))
    print("loss: " + str(loss))

    loss.backward()
    print("gradient: " + str(model.layer.weight.grad))
    
    optimizer.step()
    print("weights: " + str(model.layer.weight))
    
    optimizer.zero_grad()
    print("")

y_prob = model(x_test)

y_prob_np = torch.flatten(y_prob).detach().numpy()
y_pred_np = (y_prob_np > 0.5)
y_test_np = torch.flatten(y_test).numpy()

pred_err = np.mean(y_pred_np != y_test_np)
print("pred_err: " + str(pred_err))
