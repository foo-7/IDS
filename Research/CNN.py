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

network = Net()
