import torch
import pandas as pd
import numpy as np

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from CNN_Model import CNN_Model as CNN
from DataPreprocess import DataPreprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fileName = "Dataset_T-ITS.csv"

DP = DataPreprocess(fileName)
df = DP.runNew(targetName='class', featuresToBinary=True, targetToBinary=True)

X = df.drop(columns=['class'])
y = df['class']

print("X shape:", X.shape)
print("y shape:", y.shape)

X = df.drop(columns='class')
y = df['class']

print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)

print("X_train dtypes:\n", X_train.dtypes)
print("Unique y values:", y.unique())
print("Any NaNs in X_train?", X_train.isna().sum().sum())

print("Non-numeric columns:", X_train.select_dtypes(exclude=[float, int]).columns)

print('Train shape:', X_train.shape)
print('Validation shape:', X_val.shape)
print('Test shape:', X_test.shape)

print("Any NaNs in X_train?", X_train.isnull().any().any())
print("Any NaNs in X_val?", X_val.isnull().any().any())
print("Any NaNs in X_test?", X_test.isnull().any().any())

scaler = StandardScaler()

print("X_train type:", type(X_train))
print("X_train shape:", X_train.shape)
print("X_train head:\n", X_train.head())

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Any NaNs in scaled X_train?", np.isnan(X_train_scaled).any())
print("Min/Max of y_train:", y_train.min(), y_train.max())


X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_val = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).unsqueeze(1)

# Double Check
print("\nAfter converting to tensor and converting 2D to 3D")
print('Train shape:', X_train_tensor.shape)
print('Validation shape:', X_test_tensor.shape)
print('Test shape:', X_val_tensor.shape)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

Train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
Test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
Validation_dataset = TensorDataset(X_val_tensor, y_val_tensor)

Train_loader = DataLoader(Train_dataset, batch_size=64, shuffle=True, pin_memory=True)
Test_loader = DataLoader(Test_dataset, batch_size=64, shuffle=False, pin_memory=True)
Validation_loader = DataLoader(Validation_dataset, batch_size=64, shuffle=False, pin_memory=True)

network = CNN(input_length=X_train_tensor.shape[2]).to(device)
network.train_model(train_loader=Train_loader, validation_loader=Validation_loader, epochs=1000)
network.test_model(test_loader=Test_loader)