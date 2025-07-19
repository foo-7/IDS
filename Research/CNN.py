# Core PyTorch library for implementing deep learning models
import torch

# Provides building blocks for defining neural network layers
import torch.nn as nn

# Includes activation functions and operations used inside neural networks
import torch.nn.functional as F

# Contains optimization algorithms for training models.
import torch.optim as optim

# To handle the excel 
from torch.utils.data import Dataset

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as TTS
from DataPreprocess import DataPreprocess
from Net import Net

from torch.utils.data import TensorDataset, DataLoader

# Verifying that GPU is being utilized
print("Cuda available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
    Data preprocessing
"""
fileName1 = "jamming-merged-gps-only.csv"
fileName2 = "spoofing-merged-gps-only.csv"
DP = DataPreprocess(fileName2)
df = DP.run(givenTargets={'benign' : 0, 'malicious' : 1}, targetName='label')

X = df.drop(columns='label')
y = df['label']

X_train, X_test, y_train, y_test = TTS(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# scalar returned a NumPy array to df_scaled, so we need to make it in a DataFrame
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)


Train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
Test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

Train_loader = DataLoader(Train_dataset, batch_size=64, shuffle=True)
Test_loader = DataLoader(Test_dataset, batch_size=64, shuffle=True)

n_epochs = 3
learning_rate = 0.01
momentum = 0.5
log_interval = 10

network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

"""
TRAINING MODEL
"""

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(Train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(Train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                epoch, batch_idx * len(data), len(Train_loader), 100. * batch_idx / len(Train_loader), loss.item()
            ))
            train_losses.append(loss.item())
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len(Train_loader.dataset)))

"""
EVALUATING MODEL PERFROMANCE
"""
def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in Test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            test_loss /= len(Test_loader.dataset)
            test_losses.append(test_losses)
            print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(Test_loader.dataset), 100. * correct / len(Test_loader.dataset)
            ))

"""
Running Training and Testing Loops
"""
test()
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test