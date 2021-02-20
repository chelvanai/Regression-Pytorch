import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

iris_df = pd.read_csv('iris.csv')
x = iris_df.values[:, 0:4].astype('float32')

label_encoder = preprocessing.LabelEncoder()
iris_df['species'] = label_encoder.fit_transform(iris_df['species'])
y = iris_df['species'].values
mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(mapping)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

input_rows = x_train.shape[1]

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)


class Net(nn.Module):
    def __init__(self, input_features=4, hidden_layer1=25, hidden_layer2=30, output_features=3):
        super().__init__()
        self.fc1 = nn.Linear(input_features, hidden_layer1)
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.out = nn.Linear(hidden_layer2, output_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    out = model(x_train)
    loss = criterion(out, y_train)
    loss.backward()
    optimizer.step()

    print('number of epoch', epoch, 'loss', loss.item())

x_test = torch.from_numpy(x_test)

preds = []
with torch.no_grad():
    for val in x_test:
        y_hat = model.forward(val)
        preds.append(y_hat.argmax().item())

print(list(y_test))
print(preds)