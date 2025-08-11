import torch
import pandas as pd
import numpy as np

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from CNN_MultiClass import CNN_MultiClass as CNN
from DataPreprocess import DataPreprocess

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df_jamming = pd.read_csv('jamming-merged-gps-only.csv')
df_spoofing = pd.read_csv('spoofing-merged-gps-only.csv')

combined_columns = set()
combined_df = []

DP = DataPreprocess()
for labels in df_jamming['label'].unique():
    current_label = 0 if labels == 'benign' else 1
    current_df = df_jamming[df_jamming['label'] == labels].copy()
    current_df['label'] = current_label

    proc_df = DP.runNew(
        targetName='label',
        givenDF=current_df    
    )
    group_columns = set(proc_df.columns)
    combined_columns.update(group_columns)

    combined_df.append(proc_df.copy())

for labels in df_spoofing['label'].unique():
    current_label = 0 if labels == 'benign' else 2
    current_df = df_spoofing[df_spoofing['label'] == labels].copy()
    current_df['label'] = current_label

    proc_df = DP.runNew(
        targetName='label',
        givenDF=current_df    
    )
    group_columns = set(proc_df.columns)
    combined_columns.update(group_columns)

    combined_df.append(proc_df.copy())

aligned_df = []
for df in combined_df:
    for feature in combined_columns:
        if feature not in df.columns:
            df[feature] = 0

    cols_order = ['label'] + [c for c in combined_columns if c != 'label']
    aligned_df.append(df[cols_order])

df = pd.concat(aligned_df, ignore_index=True)
df = shuffle(df, random_state=42)

print(f'[INFO] Any duplicate values after combining and preprocessing: {df.duplicated().any()}')

corr_with_target = df.corr()['label'].drop('label').abs()
leaking_features = corr_with_target[corr_with_target > 0.9].index.tolist()

if leaking_features:
    print(f"[LEAKAGE WARNING] Dropping features correlated with target > 0.9: {leaking_features}")
    df = df.drop(columns=leaking_features)
else:
    print("[LEAKAGE CHECK] No data leakage detected.")

X = df.drop(columns='label')
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)

print(f'[DATAFRAME INFO] X_train size: {X_train.shape}')
print(f'[DATAFRAME INFO] X_test size: {X_test.shape}')
print(F'[DATAFRAME INFO] X_val size: {X_val.shape}')

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_val = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).unsqueeze(1)

print("[TORCH DATAFRAME] After converting to tensor and converting 2D to 3D")
print('[TORCH DATAFRAME] Train shape:', X_train_tensor.shape)
print('[TORCH DATAFRAME] Validation shape:', X_val_tensor.shape)
print('[TORCH DATAFRAME] Test shape:', X_test_tensor.shape)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)

Train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
Test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
Validation_dataset = TensorDataset(X_val_tensor, y_val_tensor)

Train_loader = DataLoader(Train_dataset, batch_size=64, shuffle=True, pin_memory=True)
Test_loader = DataLoader(Test_dataset, batch_size=64, shuffle=False, pin_memory=True)
Validation_loader = DataLoader(Validation_dataset, batch_size=64, shuffle=False, pin_memory=True)

network = CNN(input_length=X_train_tensor.shape[2], num_classes=len(y.unique())).to(device)
network.train_model(train_loader=Train_loader, validation_loader=Validation_loader, epochs=10)
network.test_model(test_loader=Test_loader)