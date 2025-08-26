import pandas as pd
import torch

from io import StringIO
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from models.CNN_MultiClass import CNN_MultiClass as CNN
from preprocessing.DataPreprocess import DataPreprocess

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_and_split_by_class(filename):
    dfs_by_class = defaultdict(list)

    with open(filename, 'r') as f:
        lines = f.readlines()

    header_line = None
    chunk_lines = []

    for line in lines:
        if line.startswith('timestamp_c'):
            if chunk_lines:
                chunk_csv = ''.join(chunk_lines)
                chunk_df = pd.read_csv(StringIO(chunk_csv))
                chunk_df = chunk_df.dropna(axis=1, how='all')

                if 'class' in chunk_df.columns:
                    chunk_df.rename(columns={'class': 'Label'}, inplace=True)

                chunk_df = chunk_df.loc[:, chunk_df.isna().mean() < 0.5]

                chunk_df = chunk_df.fillna(0)

                for class_label, group_df in chunk_df.groupby(chunk_df['Label'].str.lower()):
                    dfs_by_class[class_label].append(group_df)

                chunk_lines = []
            header_line = line
        chunk_lines.append(line)

    if chunk_lines:
        chunk_csv = ''.join(chunk_lines)
        chunk_df = pd.read_csv(StringIO(chunk_csv))

        chunk_df = chunk_df.dropna(axis=1, how='all')
        if 'class' in chunk_df.columns:
            chunk_df.rename(columns={'class': 'Label'}, inplace=True)
        chunk_df = chunk_df.loc[:, chunk_df.isna().mean() < 0.5]
        chunk_df = chunk_df.fillna(0)

        for class_label, group_df in chunk_df.groupby(chunk_df['Label'].str.lower()):
            dfs_by_class[class_label].append(group_df)

    concatenated = {cls: pd.concat(dfs, ignore_index=True) for cls, dfs in dfs_by_class.items()}
    return concatenated

filename = 'data/Dataset_T-ITS.csv'
dfs_by_class = load_and_split_by_class(filename)


benign_df = dfs_by_class.get('benign')
dos_df = dfs_by_class.get('dos attack')
replay_df = dfs_by_class.get('replay')
evil_twin_df = dfs_by_class.get('evil_twin')
fdi_df = dfs_by_class.get('fdi')


print(f'[TOTAL AMOUNT OF FEATURES BENIGN DATAFRAME]: {len(benign_df.columns)}')
print(f'[TOTAL AMOUNT OF FEATURES DOS DATAFRAME]: {len(dos_df.columns)}')
print(f'[TOTAL AMOUNT OF FEATURES REPLAY DATAFRAME]: {len(replay_df.columns)}')
print(f'[TOTAL AMOUNT OF FEATURES EVIL TWIN DATAFRAME]: {len(evil_twin_df.columns)}')
print(f'[TOTAL AMOUNT OF FEATURES UNIFIED DATAFRAME]: {len(fdi_df.columns)}')

print(f"[CYBER DATAFRAME INFO] ALL BELOW IS CYBER DATA AND NOT PHYSICAL DATA")
print(f"[DATAFRAME INFO] Benign shape: {benign_df.shape if benign_df is not None else 'No data'}")
print(f"[DATAFRAME INFO] DoS shape: {dos_df.shape if dos_df is not None else 'No data'}")
print(f"[DATAFRAME INFO] Replay shape: {replay_df.shape if replay_df is not None else 'No data'}")
print(f"[DATAFRAME INFO] Evil Twin shape: {evil_twin_df.shape if evil_twin_df is not None else 'No data'}")
print(f"[DATAFRAME INFO] FDI shape: {fdi_df.shape if fdi_df is not None else 'No data'}")

print(f"[DATA INFO] Total data: {benign_df.shape[0] + dos_df.shape[0] + replay_df.shape[0] + evil_twin_df.shape[0] + fdi_df.shape[0] + 5}")
print(f"[DATA INFO] Benign data amount: {benign_df.shape[0] + 1}")
print(f"[DATA INFO] All attacks amount: {dos_df.shape[0] + replay_df.shape[0] + evil_twin_df.shape[0] + fdi_df.shape[0] + 4}")
print(f'[DATA INFO] DOS amount: {dos_df.shape[0]+1}')
print(f'[DATA INFO] Replay amount: {replay_df.shape[0]+1}')
print(f'[DATA INFO] Evil Twin amount: {evil_twin_df.shape[0]+1}')
print(f'[DATA INFO] Injection amount: {fdi_df.shape[0]+1}')

all_data = [benign_df, dos_df, replay_df, evil_twin_df, fdi_df]
infoMetrics = []
label_counter = 0

for group in all_data:
    current_name = group['Label'].unique()
    group['Label'] = label_counter
    label_counter += 1
    infoMetrics.append(f"[LABEL CHANGE] Current dataframe label name: {current_name}, new label numeric name: {group['Label'].unique()}")


DP = DataPreprocess()
processed_df = []
given_columns = set()

for group in all_data:
    proc_df = DP.runNew(
        targetName='Label',
        featuresToBinary=True,
        givenDF=group
    )

    corr_with_target = proc_df.corr()['Label'].drop('Label').abs()
    leaking_features = corr_with_target[corr_with_target > 0.9].index.tolist()

    if leaking_features:
        print(f"[LEAKAGE WARNING] Dropping features correlated with target > 0.9: {leaking_features}")
    else:
        print("[LEAKAGE CHECK] No data leakage detected.")

    group_columns = set(proc_df.columns)
    given_columns.update(group_columns)
    processed_df.append(proc_df)

    infoMetrics.append(f"[INFO] Current numeric label: {group['Label'].unique()} | Prior to DP: {group.shape} | After DP: {proc_df.shape}")

for info in infoMetrics:
    print(info)

given_columns = sorted(given_columns)
aligned_df = []

for df in processed_df:
    for feature in given_columns:
        if feature not in df.columns:
            df[feature] = 0
    
    cols_order = ['Label'] + [c for c in given_columns if c != 'Label']
    aligned_df.append(df[cols_order])

final_df = pd.concat(aligned_df, ignore_index=True)
final_df = shuffle(final_df, random_state=42)

print(f"[INFO] final_df dataframe contains \'group\' column: {'group' in final_df.columns}")
print(f"[INFO] final_df dataframe contains \'Label\' column: {'Label' in final_df.columns}")
print(f"[INFO] final_df dataframe contains the following columnns:\n{given_columns}")
print(f'[FINAL DATAFRAME] Dataframe shape: {final_df.shape}')

corr_with_target = final_df.corr()['Label'].drop('Label').abs()
leaking_features = corr_with_target[corr_with_target > 0.9].index.tolist()

if leaking_features:
    print(f"[LEAKAGE WARNING] Dropping features correlated with target > 0.9: {leaking_features}")
else:
    print("[LEAKAGE CHECK] No data leakage detected.")

X = final_df.drop(columns='Label')
y = final_df['Label']

print(f'[TOTAL AMOUNT OF FEATURES UNIFIED DATAFRAME]: {len(X.columns)}')

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