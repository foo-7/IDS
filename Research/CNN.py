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
from OldNet import OldNet

from torch.utils.data import TensorDataset, DataLoader

# Verifying that GPU is being utilized
print("Cuda available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
    Data preprocessing
"""
fileName1 = "jamming-merged-gps-only.csv"  # Highest Accuracy: 78.80%
fileName2 = "spoofing-merged-gps-only.csv" # Highest Accuracy: 90.15%
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

# Since we are dealing with binary classification wth BCE loss, it should be floats and not integers
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)


Train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
Test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

Train_loader = DataLoader(Train_dataset, batch_size=64, shuffle=True)
Test_loader = DataLoader(Test_dataset, batch_size=64, shuffle=True)

n_epochs = 100
learning_rate = 0.01
momentum = 0.5
log_interval = 10

network = OldNet().to(device)
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

"""
TRAINING MODEL
"""

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(Train_loader.dataset) for i in range(n_epochs + 1)]
lowest_train_loss = float('inf')
lowest_test_loss = float('inf')
highest_test_accuracy = 0

"""
    The problem here is that since you are using
        F.nll_loss()
    it is meant for multi-class classification, where the NN would be using a softmax
    activation function for the output instead of a sigmoid function.

    The recommended use is to use:
        loss = F.binary_cross_entropy(output, target)
    since we are returning
        return torch.sigmoid(x)
    at the forward pass.

    If we were returning
        return x
    then we could use
        loss = F.binary_cross_entropy_with_logits(output, target)

    REMINDER:
        Cross Entropy is a loss function that compares:
            true labels (given as class indices or one-hot vectors)
            predicted probabilities output (vector per sample usually)

        -> Single number that tells you how "wrong" the prediction is

    EXAMPLE:
        true label (one-hot): [0, 1, 0]
        model prediction:     [0.2, 0.7, 0.2]

        cross-entropy loss is: -1 * log(0.7) = 0.357
"""
def train(epoch):
    global lowest_train_loss
    network.train()
    for batch_idx, (data, target) in enumerate(Train_loader):
        data = data.float().to(device)
        target = target.float().to(device)
        """
        Since Target shape is [64], we need to match it to the BCE of [64, 1]
        We use .view()
                -1 means -> keep the total number of elements the same
                          make the second dimension size 1 (so each element is in its own row)
                          calculate the first dimension automatically (which will be the batch size)
               second param means -> second dimension will be of size #.
        """
        target = target.view(-1,1)
            

        optimizer.zero_grad()
        output = network(data)
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if loss.item() < lowest_train_loss:
            lowest_train_loss = loss.item()

        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(Train_loader.dataset)} '
                  f'({100. * batch_idx / len(Train_loader):.0f}%)]\tLoss: {loss.item():.6f} | '
                  f'Lowest Train Loss so far: {lowest_train_loss:.6f}')

            train_losses.append(loss.item())
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len(Train_loader.dataset)))

"""
EVALUATING MODEL PERFROMANCE
"""
def test():
    global lowest_test_loss, highest_test_accuracy
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in Test_loader:
            data = data.float().to(device)
            target = target.float().to(device)

            """
            Since Target shape is [64], we need to match it to the BCE of [64, 1]
            We use .view()
                  -1 means -> keep the total number of elements the same
                              make the second dimension size 1 (so each element is in its own row)
                              calculate the first dimension automatically (which will be the batch size)
                   second param means -> second dimension will be of size #.
            """
            target = target.view(-1,1)
            
            output = network(data)

            # Debugging purposes
            #print('Output shape:', output.shape)
            #print('Target unique values:', target.unique())
            #print('Target shape:', target.shape)

            """
            By default, BCE reduction param is mean
                it averages the loss over the batch
            By using sum,
                it sums the loss over all elements instead
            """
            test_loss += F.binary_cross_entropy(input=output, target=target, reduction='sum').item()
            pred = (output > 0.5).long()
            correct += pred.eq(target.long()).sum().item()        


            # For multi-class classification
            #pred = output.data.max(1, keepdim=True)[1]
            #correct += pred.eq(target.data.view_as(pred)).sum().item()

    test_loss /= len(Test_loader.dataset)
    test_losses.append(test_losses)

    accuracy = 100. * correct / len(Test_loader.dataset)
    if test_loss < lowest_test_loss:
        lowest_test_loss = test_loss
    if accuracy > highest_test_accuracy:
        highest_test_accuracy = accuracy

    print(f'\nTest set: Avg. loss: {test_loss:.4f}, Accuracy: {correct}/{len(Test_loader.dataset)} ({accuracy:.0f}%)')
    print(f'Lowest Test Loss so far: {lowest_test_loss:.4f}')
    print(f'Highest Test Accuracy so far: {highest_test_accuracy:.2f}%\n')

"""
Running Training and Testing Loops
"""
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()