import torch
import pandas as pd
import os
import numpy as np

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from CNN_Model import CNN_Model as CNN
from DataPreprocess import DataPreprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

directory_path = 'ISOT-Drone/'
regular_files = []
attack_files = []

for subfolder in os.listdir(directory_path):
    subfolder_path = os.path.join(directory_path, subfolder)
    if os.path.isdir(subfolder_path):
        for file in os.listdir(subfolder_path):
            if file.endswith('.csv'):
                file_path = os.path.join(subfolder_path, file)
                df = pd.read_csv(file_path)
                if subfolder == 'Regular':
                    df['Label'] = 0
                    regular_files.append(df)
                else:
                    df['Label'] = 1
                    attack_files.append(df)

all_regular = pd.concat(regular_files, ignore_index=True)
all_attack = pd.concat(attack_files, ignore_index=True)

print("Regular shape:", all_regular.shape)
print("Attack shape:", all_attack.shape)

df = pd.concat([all_regular, all_attack], ignore_index=True)
print("Combined shape:", df.shape)

# Check data types and NaNs
print("Data types:\n", df.dtypes)
count = 0
for col in df.columns:
    if df[col].dtype == 'object':
        print(f"Column '{col}' is of type object with unique values: {df[col].unique()}")
        count += 1

print(f"Total object columns: {count}")
print("Any NaNs in DataFrame?", df.isnull().any().any())

DP = DataPreprocess()
df = DP.runNew(targetName='Label', givenDF=df)
print("Cleaned shape:", df.shape)

X = df.drop(columns='Label')
y = df['Label']

corr_with_target = df.corr()['Label'].drop('Label').abs()
leaking_features = corr_with_target[corr_with_target > 0.9].index.tolist()

if leaking_features:
    print(f"[LEAKAGE WARNING] Dropping features correlated with target > 0.9: {leaking_features}")
else:
    print("[LEAKAGE CHECK] No data leakage detected.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)

print("X_train type:", type(X_train))
print("X_train shape:", X_train.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_val = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).unsqueeze(1)

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
network.train_model(train_loader=Train_loader, validation_loader=Validation_loader, epochs=10)
network.test_model(test_loader=Test_loader)