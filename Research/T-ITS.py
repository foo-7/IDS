import pandas as pd
import numpy as np
import torch
import gc
import xgboost as xgb
import cupy as cp

from io import StringIO
from collections import defaultdict
from cuml.svm import SVC as cuSVC
from cuml.ensemble import RandomForestClassifier as cuRF
from torch.utils.data import TensorDataset, DataLoader
from models.CNN_MultiClass import CNN_MultiClass as CNN
from models.MLP_MultiClass import MLP_MultiClass as MLP
from preprocessing.DataPreprocess import DataPreprocess
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, precision_score

DP = DataPreprocess()
WINDOW_SIZE = 10

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
final_df = pd.concat(all_data, ignore_index=True)
final_df['Label'], mapping = pd.Series(final_df['Label']).factorize()
final_df['Source'] = (final_df.index // 500).astype(str)

del benign_df, dos_df, replay_df, evil_twin_df, fdi_df, all_data
gc.collect()

X = final_df.drop(columns=['Label', 'Source'])
y = final_df['Label']
groups = final_df['Source']

model_names = ['CNN', 'RF-SVC', 'RF', 'SVM', 'XGBOOST', 'MLP']
metric_names = ['accuracy', 'marco-f1-score', 'per-class-f1-score', 'recall', 'precision']

MODELS = {
    name: {metric: [] for metric in metric_names} 
    for name in model_names
}

gkf = StratifiedGroupKFold(n_splits=3)

"""
    Helper function to turn the flatten dataframe into sequences within each group
    for our CNN structure
    
    :param df: current dataframe
    :param window_size: current window size
"""
def create_sequences(df, window_size):
    sequences = []
    labels = []
    
    sources = df['Source'].unique()
    
    for src in sources:
        group = df[df['Source'] == src]
        
        # --- SAFETY CHECK: Skip if group is empty or smaller than window ---
        if group.empty or len(group) < window_size:
            continue
            
        features = group.drop(columns=['Label', 'Source']).values.astype(np.float32)
        
        # Only grab the target if we actually have rows
        target = group['Label'].iloc[0]

        for i in range(len(features) - window_size + 1):
            window = features[i : i + window_size]
            sequences.append(window)
            labels.append(target)
            
    # If the entire fold results in no sequences (rare but possible), 
    # handle the empty list to prevent np.array crash
    if not sequences:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int64)

    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64)

print("\n" + "="*30)
for i, name in enumerate(mapping):
    print(f'[INFO]: Label {i} corresponds to Class: {name}')
print(f'[INFO]: Total features found: {len(X.columns)}')
print(f'[INFO]: Total groups created for GKF: {len(groups.unique())}')
print("="*30 + "\n")

GLOBAL_LE = LabelEncoder()
GLOBAL_LE.fit(y)
NUM_CLASSES = len(GLOBAL_LE.classes_)
print(f"[INFO]: Encoder fitted with {NUM_CLASSES} classes.")

for fold, (train_index, test_index) in enumerate(gkf.split(X, y, groups)):
    """
    There are some del lines in case that we need to free up memory from the RAM/VRAM
    """
    print(f'\n{"="*30}[PROCESS]: STARTING FOLD {fold + 1}/3{"="*30}')

    X_train_raw, X_test_raw = X.iloc[train_index], X.iloc[test_index]
    y_train_raw, y_test_raw = y.iloc[train_index], y.iloc[test_index]

    fold_sources_train = groups.iloc[train_index]
    fold_sources_test = groups.iloc[test_index]

    # Since our CNN takes in data of a 3D Tensor: [Samples, Window_size, Features]
    # Instead of the usual 2D Matrix: [Rows, Features]
    train_df_for_seq = pd.concat([X_train_raw, y_train_raw], axis=1)

    train_cleaned = DP.runNew(targetName='Label', givenDF=train_df_for_seq)
    print('[INFO]: Preprocessing finished')
    source_map = fold_sources_train.to_dict()
    train_cleaned['Source'] = train_cleaned.index.map(source_map)
    train_cleaned = train_cleaned.reset_index(drop=True)

    test_df_for_seq = pd.concat([X_test_raw, y_test_raw], axis=1)
    test_df_for_seq['Source'] = fold_sources_test.values
    test_df_for_seq = test_df_for_seq.reset_index(drop=True)

    del fold_sources_train, fold_sources_test, source_map
    gc.collect()

    selected_features = train_cleaned.drop(columns=['Label', 'Source']).columns

    scaler = StandardScaler()
    train_cleaned[selected_features] = scaler.fit_transform(train_cleaned[selected_features])

    # Data contains NaNs, need to remove them
    X_test_raw = X_test_raw.replace([np.inf, -np.inf], np.nan).dropna(subset=selected_features)
    y_test_raw = y_test_raw.loc[X_test_raw.index]
    fold_sources_test = groups.iloc[test_index].loc[X_test_raw.index]

    X_test_scaled_array = scaler.transform(X_test_raw[selected_features])

    train_features_only = train_cleaned[selected_features]
    test_features_only = pd.DataFrame(X_test_scaled_array, columns=selected_features)
    audit_overlap = pd.merge(train_features_only, test_features_only, how='inner')
    print(f'[AUDIT]: Samples overlapping across split: {len(audit_overlap)}')
    if len(audit_overlap) > 0:
        print(f'[AUDIT WARNING]: {len(audit_overlap)} test samples are identical to training samples.')
    else:
        print(f'[AUDIT PASSED]: Training and Testing sets are distinct')
    
    # Our data for RF-SVC/RF/SVM/XGBOOST
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(train_cleaned[selected_features])
    X_test_pca = pca.transform(pd.DataFrame(X_test_scaled_array, columns=selected_features))
    print(f'[PCA INFO]: Reduced features from {len(selected_features)} to {pca.n_components_}')


    X_train_gpu = cp.array(X_train_pca)
    y_train_gpu = cp.array(train_cleaned['Label'].values)
    y_train_cpu = train_cleaned['Label'].values

    X_test_gpu = cp.array(X_test_pca)
    y_test_gpu = cp.array(y_test_raw.values)
    y_test_cpu = y_test_raw.values

    y_train_cpu = GLOBAL_LE.transform(y_train_cpu)
    y_train_gpu = cp.array(y_train_cpu)
    y_test_cpu = GLOBAL_LE.transform(y_test_cpu)
    y_test_gpu = cp.array(y_test_cpu)

    fold_labels_combined = np.concatenate([y_train_cpu, y_test_cpu])
    
    local_le = LabelEncoder()
    y_train_xgb = local_le.fit_transform(y_train_cpu) # Maps [0, 2, 4, 6, 9] -> [0, 1, 2, 3, 4]
    y_train_gpu_xgb = cp.array(y_train_xgb)
    mask = np.isin(y_test_cpu, local_le.classes_)
    X_test_gpu_filtered = X_test_gpu[cp.array(mask)]
    y_test_cpu_filtered = y_test_cpu[mask]
    y_test_xgb = local_le.transform(y_test_cpu_filtered)
    y_test_gpu_xgb = cp.array(y_test_xgb)
    fold_num_classes = len(local_le.classes_)

    # Data for CNN/MLP
    window_size = WINDOW_SIZE
    X_train_seq, y_train_seq = create_sequences(train_cleaned, window_size=window_size)
    y_train_seq = GLOBAL_LE.transform(y_train_seq)

    # Save RAM
    X_train_seq = X_train_seq.astype(np.float32)
    y_train_seq = y_train_seq.astype(np.int64)

    test_aligned = pd.DataFrame(X_test_scaled_array, columns=selected_features, index=X_test_raw.index)
    test_aligned['Label'] = y_test_raw.values
    test_aligned['Source'] = fold_sources_test.values
    test_aligned = test_aligned.reset_index(drop=True)
    X_test_seq, y_test_seq = create_sequences(test_aligned, window_size=window_size)
    y_test_seq = GLOBAL_LE.transform(y_test_seq)

    # Same principle here
    X_test_seq = X_test_seq.astype(np.float32)
    y_test_seq = y_test_seq.astype(np.int64)
    
    print(f'[INFO]: 2D Train Shape: {X_train_pca.shape}')
    print(f'[INFO]: 3D CNN Train Shape: {X_train_seq.shape}')

    X_train_final, X_val_seq, y_train_final, y_val_seq = train_test_split(
        X_train_seq, y_train_seq, test_size=0.15, random_state=42
    )

    X_train_tensor = torch.from_numpy(X_train_final).float()
    y_train_tensor = torch.from_numpy(y_train_final).long()

    X_val_tensor = torch.from_numpy(X_val_seq).float()
    y_val_tensor = torch.from_numpy(y_val_seq).long()

    X_test_tensor = torch.from_numpy(X_test_seq).float()
    y_test_tensor = torch.from_numpy(y_test_seq).long()

    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    val_ds = TensorDataset(X_val_tensor, y_val_tensor)
    test_ds = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    del train_df_for_seq, test_df_for_seq, train_cleaned, test_aligned
    del train_features_only, test_features_only, audit_overlap
    del X_train_final, X_val_seq, y_train_final, y_val_seq
    del X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor
    gc.collect()

    """
    CLASSICAL MODELS (RF, SVM, XGBOOST, RF-SVC)
    """
    # RF-SVC
    RF = xgb.XGBRFClassifier(
        objective='multi:softprob',
        num_class=fold_num_classes,
        n_estimators=100,
        random_state=42,
        tree_method='hist',
        device='cuda'
    )

    # Using cross_val_predict ensures the RF only predicts on data it has not seen
    # this prevents our SVC from cheating by learning from RF's memorization
    RF_train_features = cross_val_predict(RF, X_train_pca, y_train_xgb, cv=3, method='predict_proba')
    RF_train_features_gpu = cp.array(RF_train_features)

    SVC = cuSVC(kernel='rbf', probability=True)
    SVC.fit(RF_train_features_gpu, y_train_gpu_xgb)

    # Re-fit the RF on total training fold data
    RF.fit(X_train_gpu, y_train_gpu_xgb)
    RF_test_features = RF.predict_proba(X_test_gpu_filtered)
    hybrid_preds = SVC.predict(RF_test_features)
    
    y_pred_cpu = cp.asnumpy(hybrid_preds)
    hybrid_acc = balanced_accuracy_score(y_test_xgb, y_pred_cpu)
    macrof1 = f1_score(y_test_xgb, y_pred_cpu, average='macro')
    recall = recall_score(y_test_xgb, y_pred_cpu, average='macro')
    prec = precision_score(y_test_xgb, y_pred_cpu, average='macro')
    print(f'\n{"="*30}[HYBRID RF-SVC METRICS]{"="*30}')
    print(f'[RESULT | ACCURACY]: Hybrid RF-SVC Accuracy: {hybrid_acc:.4f}')
    print(f'[RESULT | MACRO F1-SCORE]: Hybrid RF-SVC Macro F1-score: {macrof1:.4f}')
    print(f'[RESULT | RECALL]: Hybrid RF-SVC Recall Score: {recall:.4f}')
    print(f'[RESULT | PREICSION]: Hybrid RF-SVC Precision Score: {prec:.4f}')
    MODELS['RF-SVC']['accuracy'].append(hybrid_acc)
    MODELS['RF-SVC']['macro-f1-score'].append(macrof1)
    MODELS['RF-SVC']['recall'].append(recall)
    MODELS['RF-SVC']['precision'].append(prec)

    del RF, SVC
    del RF_train_features, RF_train_features_gpu, RF_test_features
    del y_pred_cpu, hybrid_preds, hybrid_acc, macrof1, recall, prec
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()

    # RF
    RF = cuRF(n_estimators=100)
    RF.fit(X_train_gpu, y_train_gpu_xgb)
    pred = RF.predict(X_test_gpu_filtered)
    y_pred_cpu = cp.asnumpy(pred)
    acc = balanced_accuracy_score(y_test_xgb, y_pred_cpu)
    macrof1 = f1_score(y_test_xgb, y_pred_cpu, average='macro')
    recall = recall_score(y_test_xgb, y_pred_cpu, average='macro')
    prec = precision_score(y_test_xgb, y_pred_cpu, average='macro')
    print(f'\n{"="*30}[RANDOM FOREST METRICS]{"="*30}')
    print(f'[RESULT | ACCURACY]: RANDOM FOREST Accuracy: {acc:.4f}')
    print(f'[RESULT | MACRO F1-SCORE]: RANDOM FOREST Macro F1-score: {macrof1:.4f}')
    print(f'[RESULT | RECALL]: RANDOM FOREST Recall Score: {recall:.4f}')
    print(f'[RESULT | PREICSION]: RANDOM FOREST Precision Score: {prec:.4f}')
    MODELS['RF']['accuracy'].append(acc)
    MODELS['RF']['macro-f1-score'].append(macrof1)
    MODELS['RF']['recall'].append(recall)
    MODELS['RF']['precision'].append(prec)

    del RF, pred, acc, y_pred_cpu, macrof1, recall, prec
    gc.collect()

    # SVM
    SVC = cuSVC(kernel='rbf')
    SVC.fit(X_train_gpu, y_train_gpu_xgb)
    pred = SVC.predict(X_test_gpu)
    y_pred_cpu = cp.asnumpy(pred)
    acc = balanced_accuracy_score(y_test_xgb, y_pred_cpu)
    macrof1 = f1_score(y_test_xgb, y_pred_cpu, average='macro')
    recall = recall_score(y_test_xgb, y_pred_cpu, average='macro')
    prec = precision_score(y_test_xgb, y_pred_cpu, average='macro')
    print(f'\n{"="*30}[SUPPORT VECTOR MACHINE METRICS]{"="*30}')
    print(f'[RESULT | ACCURACY]: SUPPORT VECTOR MACHINE Accuracy: {acc:.4f}')
    print(f'[RESULT | MACRO F1-SCORE]: SUPPORT VECTOR MACHINE Macro F1-score: {macrof1:.4f}')
    print(f'[RESULT | RECALL]: SUPPORT VECTOR MACHINE Recall Score: {recall:.4f}')
    print(f'[RESULT | PREICSION]: SUPPORT VECTOR MACHINE Precision Score: {prec:.4f}')
    MODELS['SVM']['accuracy'].append(acc)
    MODELS['SVM']['macro-f1-score'].append(macrof1)
    MODELS['SVM']['recall'].append(recall)
    MODELS['SVM']['precision'].append(prec)

    del SVC, pred, acc, y_pred_cpu, acc, macrof1, recall, prec
    gc.collect()

    # XGBOOST
    XGBOOST = xgb.XGBClassifier(tree_method='hist', device='cuda')
    XGBOOST.fit(X_train_gpu, y_train_gpu_xgb)
    pred = XGBOOST.predict(X_test_gpu)
    y_pred_cpu = cp.asnumpy(pred)
    acc = balanced_accuracy_score(y_test_xgb, y_pred_cpu)
    macrof1 = f1_score(y_test_xgb, y_pred_cpu, average='macro')
    recall = recall_score(y_test_xgb, y_pred_cpu, average='macro')
    prec = precision_score(y_test_xgb, y_pred_cpu, average='macro')
    print(f'\n{"="*30}[XGBOOSTC METRICS]{"="*30}')
    print(f'[RESULT | ACCURACY]: XGBOOST Accuracy: {acc:.4f}')
    print(f'[RESULT | MACRO F1-SCORE]: XGBOOST Macro F1-score: {macrof1:.4f}')
    print(f'[RESULT | RECALL]: XGBOOST Recall Score: {recall:.4f}')
    print(f'[RESULT | PREICSION]: XGBOOST Precision Score: {prec:.4f}')
    MODELS['XGBOOST']['accuracy'].append(acc)
    MODELS['XGBOOST']['macro-f1-score'].append(macrof1)
    MODELS['XGBOOST']['recall'].append(recall)
    MODELS['XGBOOST']['precision'].append(prec)
    
    del XGBOOST, pred, acc, y_pred_cpu, acc, macrof1, recall, prec
    gc.collect()

    print('[PROCESS]: Classical models trained and tested. Freeing memory for CNN and MLP')
    del X_train_pca, X_test_pca, X_test_scaled_array
    del X_train_gpu, X_test_gpu, y_train_gpu, y_test_gpu
    del y_train_cpu
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()

    """
    DEEP LEARNING MODELS (CNN, MLP)
    """
    # CNN
    fold_path = f'best_weights/CNN_ISOTDrone_fold_{fold+1}.pth'
    CNN_model = CNN(num_features=len(selected_features), window_size=WINDOW_SIZE, num_classes=NUM_CLASSES)
    CNN_model.train_model(train_loader=train_loader, validation_loader=val_loader, epochs=20, path=fold_path)
    RESULTS = CNN_model.test_model(test_loader=test_loader, path=fold_path)
    MODELS['CNN']['accuracy'].append(RESULTS['accuracy'])
    MODELS['CNN']['macro-f1-score'].append(RESULTS['f1'])
    MODELS['CNN']['recall'].append(RESULTS['recall'])
    MODELS['CNN']['precision'].append(RESULTS['precision'])

    
    del CNN_model, RESULTS, fold_path
    torch.cuda.empty_cache()
    gc.collect()

    # MLP (2-3 Dense Layers)
    fold_path = f'best_weights/MLP_ISOTDrone_fold_{fold+1}.pth'
    MLP_model = MLP(num_features=len(selected_features), window_size=WINDOW_SIZE, num_classes=NUM_CLASSES)
    MLP_model.train_model(train_loader=train_loader, validation_loader=val_loader, epochs=20, path=fold_path)
    RESULTS = MLP_model.test_model(test_loader=test_loader, path=fold_path)
    MODELS['MLP']['accuracy'].append(RESULTS['accuracy'])
    MODELS['MLP']['macro-f1-score'].append(RESULTS['f1'])
    MODELS['MLP']['recall'].append(RESULTS['recall'])
    MODELS['MLP']['precision'].append(RESULTS['precision'])

    del MLP_model, RESULTS, fold_path
    del train_loader, test_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

results_summary = []
for model_name, metrics in MODELS.items():
    row = {'Model': model_name}
    for m_name, values in metrics.items():
        if values:
            row[m_name] = f"{np.mean(values):.4f} ± {np.std(values):.4f}"
    results_summary.append(row)

report_df = pd.DataFrame(results_summary)
print("\n" + "="*50 + "\nFINAL CROSS-VALIDATION RESULTS\n" + "="*50)
print(report_df.to_string(index=False))