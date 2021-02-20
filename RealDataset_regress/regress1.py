import numpy as np
from torch import nn
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.nn.functional as F

df = pd.read_csv('kc_house_data.csv')

x = df[['bedrooms', 'bathrooms', 'waterfront', 'sqft_living', 'sqft_lot', 'floors', 'grade']]
y = df["price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
input_rows = x_train.shape[1]

x_train = torch.from_numpy(x_train.values.astype(np.float32))
y_train = torch.from_numpy(y_train.values.astype(np.float32))


class Net(nn.Module):
    def __init__(self, input_features=input_rows, hidden_layer1=5, hidden_layer2=10, output_features=1):
        super().__init__()
        self.fc1 = nn.Linear(input_features, hidden_layer1)
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.out = nn.Linear(hidden_layer1, output_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


model = Net()

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
