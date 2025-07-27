import torch
import pandas as pd

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from CNN_Model import CNN_Model as CNN
from DataPreprocess import DataPreprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fileName1 = "jamming-merged-gps-only.csv"
fileName2 = "spoofing-merged-gps-only.csv"

DP = DataPreprocess(fileName=fileName2)
df = DP.run(
    givenTargets={
        'benign' : 0,
        'malicious' : 1
    },
    targetName='label'
)

X = df.drop(columns='label')
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)

print('Train shape:', X_train.shape) # shape (972, 62)
print('Validation shape:', X_val.shape) # shape (324, 62)
print('Test shape:', X_test.shape) # shape (325, 62)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_val = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

# Since we are dealing with binary classification wth BCE loss, it should be floats and not integers
# Unsqueezing to shape into 3D for convolution layers
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).unsqueeze(1)  # shape (972, 1, 62)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).unsqueeze(1)   # shape (325, 1, 62)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).unsqueeze(1) # shape (325, 1, 62)

# Double Check
print("\nAfter converting to tensor and converting 2D to 3D")
print('Train shape:', X_train_tensor.shape) # shape (972, 1, 62)
print('Validation shape:', X_test_tensor.shape) # shape (324, 1, 62)
print('Test shape:', X_val_tensor.shape) # shape (325, 1, 62)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

Train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
Test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
Validation_dataset = TensorDataset(X_val_tensor, y_val_tensor)

# Shuffle train data but not test and validation for consistent evaluation
Train_loader = DataLoader(Train_dataset, batch_size=64, shuffle=True, pin_memory=True)
Test_loader = DataLoader(Test_dataset, batch_size=64, shuffle=False, pin_memory=True)
Validation_loader = DataLoader(Validation_dataset, batch_size=64, shuffle=False, pin_memory=True)

network = CNN(input_length=X_train_tensor.shape[2]).to(device)
network.train_model(train_loader=Train_loader, validation_loader=Validation_loader, epochs=1000)
network.test_model(test_loader=Test_loader)