import numpy as np
from torch import nn
import torch
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('house-price-room&hall&bathroom.csv')

x = df[['rooms','waterfall']]
y = df["price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
input_rows = x_train.shape[1]

x_train = torch.from_numpy(x_train.values.astype(np.float32))
y_train = torch.from_numpy(y_train.values.astype(np.float32))


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(input_rows, 1)  # One in and one out

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = Model()
print(model.linear.weight)

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    # 1) Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_train)

    # 2) Compute and print loss
    loss = criterion(y_pred, y_train.view(y_train.shape[0], 1))
    print(f'Epoch: {epoch} | Loss: {loss.item()} ')

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(x_train)
print(y_train.view(y_train.shape[0], 1))
