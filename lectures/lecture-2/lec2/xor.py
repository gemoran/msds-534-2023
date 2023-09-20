

##
##
## A simpler version of XOR neural net
## from lecture 2
##
## - ReLU activation for hidden layer
## - no mini-batching
## - allows for inspection of the learned weights
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


n = 2000
p = 2
x = np.random.uniform(-2, 2, size=(n, p))

##beta0 = np.array([0.5, 0.5])
##probs = np.exp(x @ beta0) / (1 + np.exp(x @ beta0))
##y = np.random.binomial(1, probs)

y = ((x[:,0] < 0) & (x[:, 1] > 0)).astype(x.dtype) + ((x[:,0] > 0) & (x[:, 1] < 0)).astype(x.dtype)

x = torch.tensor(x, dtype=torch.float)
y = torch.tensor(y, dtype=torch.float).view(-1, 1)


ntrain = 1500
x_train = x[:ntrain]
y_train = y[:ntrain]
x_test = x[ntrain:]
y_test = y[ntrain:]

class LRModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, out_dim):
        super(LRModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.ReLU()(x)
        x = self.layer2(x)
        x = nn.Sigmoid()(x)

        return(x)
    
hidden_dim = 4

model = LRModel(x.shape[1], hidden_dim, 1)
optimizer = optim.SGD(model.parameters(), lr=0.1)

pred = model(x_train)

def loss_fn(y_pred, y_true):
    loss_vec = y_true * torch.log(y_pred + 1e-8) + (1 - y_true) * torch.log(1 - y_pred + 1e-8)
    output = -1.0 * torch.mean(loss_vec)
    return(output)

num_iter = 1000

for i in range(num_iter):
    ##loss = nn.BCELoss()(model(x_train), y_train)
    loss = loss_fn(model(x_train), y_train.view(-1, 1))

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


print("layer1 weights: " + str(model.layer1.weight))
print("layer2 weights: " + str(model.layer2.weight))

y_pred = model(x_test)
y_pred = np.where(y_pred > 0.5, 1, 0)
y_pred = torch.tensor(y_pred, dtype=torch.float).view(-1,1)

test_err = abs(y_pred - y_test).mean()
print("test error: " + str(test_err))


## visualization

x1 = np.arange(-2, 2, 0.05)
x2 = np.arange(-2, 2, 0.05)

x_test_np = np.array([(i, j) for i in x1 for j in x2])
y_test_np = ((x_test_np[:,0] < 0) & (x_test_np[:, 1] > 0)).astype(x_test_np.dtype) + ((x_test_np[:,0] > 0) & (x_test_np[:, 1] < 0)).astype(x_test_np.dtype)

x_test = torch.tensor(x_test_np, dtype=torch.float)
y_test = torch.tensor(y_test_np)

y_pred = model(x_test)

y_pred_np = y_pred.detach().numpy()
y_pred_np = y_pred_np.reshape(x1.shape[0], x2.shape[0])

seaborn_cols = sns.color_palette("tab10")
cols = [seaborn_cols[int(i)] for i in y]

custom_cmap = sns.diverging_palette(220, 50, s=70, l=70, as_cmap=True)

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(4, 4)
ax.contourf(x1, x2, y_pred_np, cmap=custom_cmap)
ax.scatter(x[0:100,0], x[0:100,1], c=cols[0:100])
fig.savefig('tmp_xor_nn_d_' + str(hidden_dim) + '.png')