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


df = pd.read_csv('Dataset_T-ITS.csv')

print(f'[INFO] The current dataframe columns:\n{df.columns}')
print(f'[INFO] Current unique labels found in the dataframe: {df['class'].unique()}')
print(f'[INFO] The amount of unique labels found: {len(df['class'].unique())}')

label_counter = 0
mapped_df = []
infoMetrics = []
debug_mapping = {}
found_columns = set()
old_df_value = df.shape

DP = DataPreprocess()
for label in df['class'].unique():
    current_df = df[df['class'].astype(str) == label].copy()
    if not current_df.empty:
        current_df['class'] = label_counter
        proc_df = DP.runNew(
            targetName='class',
            featuresToBinary=True,
            givenDF=current_df
        )

        if proc_df.shape[0] >= 2:
            debug_mapping[label] = label_counter
            group_columns = set(proc_df.columns)
            found_columns.update(group_columns)
            label_counter += 1
            mapped_df.append(proc_df.copy())
            infoMetrics.append(f'[INFO] Current numeric label: {label_counter - 1} | Prior to DP: {current_df.shape} | After DP: {proc_df.shape}')
        else:
            infoMetrics.append(f'[INFO] Current numeric label: {label_counter - 1} | Cannot append to dataframe due to not enough data: {current_df.shape}')
    
    else:
        infoMetrics.append(f'[WARNING] Current label: {label} | If current label is NaN, then do not worry.')

print(f'[DEBUGGING] Label mapping: {debug_mapping}')

aligned_df = []
for df in mapped_df:
    for feature in found_columns:
        if feature not in df.columns:
            df[feature] = 0
    
    cols_order = ['class'] + [c for c in found_columns if c != 'class']
    aligned_df.append(df[cols_order])

df = pd.concat(mapped_df, ignore_index=True)
df = shuffle(df, random_state=42)
print(f'[INFO] New mapping for labels in df: {df["class"].unique()}')
corr_with_target = df.corr()['class'].drop('class').abs()
leaking_features = corr_with_target[corr_with_target > 0.9].index.tolist()

if leaking_features:
    print(f"[LEAKAGE WARNING] Dropping features correlated with target > 0.9: {leaking_features}")
else:
    print("[LEAKAGE CHECK] No data leakage detected.")

for info in infoMetrics:
    print(info)

X = df.drop(columns='class')
y = df['class']
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