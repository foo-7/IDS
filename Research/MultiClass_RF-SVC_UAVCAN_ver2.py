import pandas as pd
import glob
import numpy as np
import cupy as cp
import xgboost as xgb
from cuml.svm import SVC as cuSVC

from preprocessing.DataPreprocess import DataPreprocess
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics

bin_files = glob.glob('data/UAVCAN-Attack/*.bin')
file_size = {}
column_names = [
    'Status', 'Time', 'Interface', 'ID', 'Length', 'Data1', 'Data2', 
    'Data3', 'Data4', 'Data5', 'Data6', 'Data7','Data8'
]

df_files = []
for file in bin_files:
    df = pd.read_csv(file, sep=r'\s+', names=column_names)
    df['Time'] = df['Time'].str.replace('[()]', '', regex=True)  # Remove parentheses
    df['Time'] = df['Time'].astype(float)
    file_size[file] = df.shape[0]
    df_files.append(df)

print(len(df_files) == 10)

"""
    NORMAL : 0
    FLOODING : 1
    FUZZY : 2
    REPLAY : 3
"""

malicious_type1 = (df_files[1]['Status'] == 'Attack').sum() # Flooding
malicious_type2 = (df_files[2]['Status'] == 'Attack').sum() # Flooding
malicious_type3 = (df_files[3]['Status'] == 'Attack').sum() # Fuzzy
malicious_type4 = (df_files[4]['Status'] == 'Attack').sum() # Fuzzy
malicious_type5 = (df_files[5]['Status'] == 'Attack').sum() # Replay
malicious_type6 = (df_files[6]['Status'] == 'Attack').sum() # Replay

# Flooding
for index, row in df_files[1].iterrows():
    if row['Status'] == 'Normal': df_files[1].at[index, 'Status'] = 0
    else: df_files[1].at[index, 'Status'] = 1

# Flooding
for index, row in df_files[2].iterrows():
    if row['Status'] == 'Normal': df_files[2].at[index, 'Status'] = 0
    else: df_files[2].at[index, 'Status'] = 1

# Fuzzy
for index, row in df_files[3].iterrows():
    if row['Status'] == 'Normal': df_files[3].at[index, 'Status'] = 0
    else: df_files[3].at[index, 'Status'] = 2

# Fuzzy
for index, row in df_files[4].iterrows():
    if row['Status'] == 'Normal': df_files[4].at[index, 'Status'] = 0
    else: df_files[4].at[index, 'Status'] = 2

# Replay
for index, row in df_files[5].iterrows():
    if row['Status'] == 'Normal': df_files[5].at[index, 'Status'] = 0
    else: df_files[5].at[index, 'Status'] = 3

# Replay
for index, row in df_files[6].iterrows():
    if row['Status'] == 'Normal': df_files[6].at[index, 'Status'] = 0
    else: df_files[6].at[index, 'Status'] = 3

malicious_type7_Flooding = 0
malicious_type7_Fuzzy = 0
found_attack = 0
for index, row in df_files[7].iterrows():
    if row['Status'] == 'Normal': df_files[7].at[index, 'Status'] = 0

    time_rounded = round(row['Time'])
    # Flooding
    if time_rounded >= 48 and \
       time_rounded <= 92 and \
       row['Status'] == 'Attack':
        malicious_type7_Flooding += 1
        df_files[7].at[index, 'Status'] = 1
    
    # Fuzzy
    elif time_rounded >= 98 and \
         time_rounded <= 132 and \
         row['Status'] == 'Attack':
        malicious_type7_Fuzzy += 1
        df_files[7].at[index, 'Status'] = 2

    # Flooding
    elif time_rounded >= 138 and \
         time_rounded <= 182 and \
         row['Status'] == 'Attack':
        malicious_type7_Flooding += 1
        df_files[7].at[index, 'Status'] = 1

    # Fuzzy
    elif time_rounded >= 188 and \
         time_rounded <= 222 and \
         row['Status'] == 'Attack':
        malicious_type7_Fuzzy += 1
        df_files[7].at[index, 'Status'] = 2

malicious_type8_Replay = 0
malicious_type8_Fuzzy = 0
for index, row in df_files[8].iterrows():
    if row['Status'] == 'Normal': df_files[8].at[index, 'Status'] = 0
    time_rounded = round(row['Time'])

    # Fuzzy
    if time_rounded >= 60 and \
       time_rounded <= 100 and \
       row['Status'] == 'Attack':
        malicious_type8_Fuzzy += 1
        df_files[8].at[index, 'Status'] = 2
    
    # Replay
    elif time_rounded >= 110 and \
         time_rounded <= 140 and \
         row['Status'] == 'Attack':
        malicious_type8_Replay += 1
        df_files[8].at[index, 'Status'] = 3

    # Fuzzy
    elif time_rounded >= 150 and \
         time_rounded <= 190 and \
         row['Status'] == 'Attack':
        malicious_type8_Fuzzy += 1
        df_files[8].at[index, 'Status'] = 2

    # Replay
    elif time_rounded >= 200 and \
         time_rounded <= 230 and \
         row['Status'] == 'Attack':
        malicious_type8_Replay += 1
        df_files[8].at[index, 'Status'] = 3

malicious_type9_Flooding = 0
malicious_type9_Replay = 0
for index, row in df_files[9].iterrows():
    if row['Status'] == 'Normal': df_files[9].at[index, 'Status'] = 0
    time_rounded = round(row['Time'])

    # Flooding
    if time_rounded >= 55 and \
       time_rounded <= 114 and \
       row['Status'] == 'Attack':
        malicious_type9_Flooding += 1
        df_files[9].at[index, 'Status'] = 1
    
    # Replay
    elif time_rounded >= 115 and \
         time_rounded <= 154 and \
         row['Status'] == 'Attack':
        malicious_type9_Replay += 1
        df_files[9].at[index, 'Status'] = 3

    # Flooding
    elif time_rounded >= 155 and \
         time_rounded <= 204 and \
         row['Status'] == 'Attack':
        malicious_type9_Flooding += 1
        df_files[9].at[index, 'Status'] = 1

    # Replay
    elif time_rounded >= 205 and \
         time_rounded <= 270 and \
         row['Status'] == 'Attack':
        malicious_type9_Replay += 1
        df_files[9].at[index, 'Status'] = 3

malicious_type10_Flooding = 0
malicious_type10_Fuzzy = 0
malicious_type10_Replay = 0
for index, row in df_files[0].iterrows():
    if row['Status'] == 'Normal': df_files[0].at[index, 'Status'] = 0
    time_rounded = round(row['Time'])

    # Flooding
    if time_rounded >= 60 and \
       time_rounded <= 110 and \
       row['Status'] == 'Attack':
        malicious_type10_Flooding += 1
        df_files[0].at[index, 'Status'] = 1
    
    # Fuzzy
    elif time_rounded >= 120 and \
         time_rounded <= 160 and \
         row['Status'] == 'Attack':
        malicious_type10_Fuzzy += 1
        df_files[0].at[index, 'Status'] = 2

    # Replay
    elif time_rounded >= 170 and \
         time_rounded <= 200 and \
         row['Status'] == 'Attack':
        malicious_type10_Replay += 1
        df_files[0].at[index, 'Status'] = 3

print(f"[TYPE 1]: Flooding attacks amount: {malicious_type1} | Total attacks: {malicious_type1}")
print(f"[TYPE 2]: Flooding attacks amount: {malicious_type2} | Total attacks: {malicious_type2}")
print(f"[TYPE 3]: Fuzzy attacks amount: {malicious_type3} | Total attacks: {malicious_type3}")
print(f"[TYPE 4]: Fuzzy attacks amount: {malicious_type4} | Total attacks: {malicious_type4}")
print(f"[TYPE 5]: Replay attacks amount: {malicious_type5} | Total attacks: {malicious_type5}")
print(f"[TYPE 6]: Replay attacks amount: {malicious_type6} | Total attacks: {malicious_type6}")
print(f"[TYPE 7]: Flooding attacks amount: {malicious_type7_Flooding}, Fuzzy attacks amount: {malicious_type7_Fuzzy} | Total attacks: {malicious_type7_Flooding + malicious_type7_Fuzzy}")
print(f"[TYPE 8]: Fuzzy attacks amount: {malicious_type8_Fuzzy}, Replay attacks amount: {malicious_type8_Replay} | Total attacks: {malicious_type8_Fuzzy + malicious_type8_Replay}")
print(f"[TYPE 9]: Flooding attacks amount: {malicious_type9_Flooding}, Replay attacks amount: {malicious_type9_Replay} | Total attacks: {malicious_type9_Flooding + malicious_type9_Replay}")
print(f"[TYPE 10]: Flooding attacks amount: {malicious_type10_Flooding}, Fuzzy attacks amount: {malicious_type10_Fuzzy}, Replay attacks amount: {malicious_type10_Replay} | Total attacks: {malicious_type10_Flooding+malicious_type10_Replay+malicious_type10_Fuzzy}")

total_flooding = malicious_type1 + malicious_type2 + malicious_type7_Flooding + malicious_type9_Flooding + malicious_type10_Flooding
total_fuzzy = malicious_type3 + malicious_type4 + malicious_type8_Fuzzy + malicious_type7_Fuzzy + malicious_type10_Fuzzy
total_replay = malicious_type5 + malicious_type6 + malicious_type8_Replay + malicious_type9_Replay + malicious_type10_Replay
print(f"[ATTACK INFO] Total flooding: {total_flooding} | Total fuzzy: {total_fuzzy} | Total replay: {total_replay}")
print(
    f"[DOUBLE CHECK] Type 1: {(df_files[1]['Status'] == 'Attack').sum()} == {malicious_type1} | Type 2: {(df_files[2]['Status'] == 'Attack').sum()} == {malicious_type2} | Type 3: {(df_files[3]['Status'] == 'Attack').sum()}  == {malicious_type3}" + \
    f"| Type 4: {(df_files[4]['Status'] == 'Attack').sum()} == {malicious_type4} | Type 5: {(df_files[5]['Status'] == 'Attack').sum()} == {malicious_type5} | Type 6: {(df_files[6]['Status'] == 'Attack').sum()} == {malicious_type6} " + \
    f"| Type 7: {(df_files[7]['Status'] == 'Attack').sum()} == {malicious_type7_Flooding + malicious_type7_Fuzzy} | Type 8: {(df_files[8]['Status'] == 'Attack').sum()} == {malicious_type8_Fuzzy + malicious_type8_Replay} | Type 9: {(df_files[9]['Status'] == 'Attack').sum()} == {malicious_type9_Flooding + malicious_type9_Replay} " + \
    f"| Type 10: {(df_files[0]['Status'] == 'Attack').sum()} == {malicious_type10_Flooding + malicious_type10_Fuzzy + malicious_type10_Replay}"
)

print(f"[TARGET DATATYPE] Status label for the following 10 datasets:\n{df_files[1]['Status'].dtype}\n{df_files[2]['Status'].dtype}\n{df_files[3]['Status'].dtype}\n{df_files[4]['Status'].dtype}\n{df_files[5]['Status'].dtype}\n{df_files[6]['Status'].dtype}\n{df_files[7]['Status'].dtype}\n{df_files[8]['Status'].dtype}\n{df_files[9]['Status'].dtype}\n{df_files[0]['Status'].dtype}")

for i in range(len(df_files)):
    df_files[i]['Status'] = pd.to_numeric(df_files[i]['Status'], errors='coerce')

processed_df_files = []
given_columns = set()
infoMetrics = []

DP = DataPreprocess()
current_index = 0
for df in df_files:
    processed_df = DP.runNew(
        targetName='Status',
        featuresToBinary=True,
        givenDF=df
    )
    processed_df_files.append(processed_df)
    group_columns = set(processed_df.columns)
    given_columns.update(group_columns)

    infoMetrics.append(f"[INFO] Current index: {current_index} | Prior to DP: {df.shape} | After DP: {processed_df.shape}")
    current_index += 1

for info in infoMetrics:
    print(info)

given_columns = sorted(given_columns)
aligned_df = []

for df in processed_df_files:
    for feature in given_columns:
        if feature not in df.columns:
            df[feature] = 0
    
    cols_order = ['Status'] + [c for c in given_columns if c != 'Status']
    aligned_df.append(df[cols_order])

final_df = pd.concat(aligned_df, ignore_index=True)
final_df = shuffle(final_df, random_state=42)

print(f'[FINAL DATAFRAME] Dataframe shape: {final_df.shape}')

X = final_df.drop(columns='Status')
y = final_df['Status']

corr_with_target = final_df.corr()['Status'].drop('Status').abs()
leaking_features = corr_with_target[corr_with_target > 0.9].index.tolist()

if leaking_features:
    print(f"[LEAKAGE WARNING] Dropping features correlated with target > 0.9: {leaking_features}")
else:
    print("[LEAKAGE CHECK] No data leakage detected.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=0.95)
X_train_reduced = pca.fit_transform(X_train_scaled)
X_test_reduced = pca.transform(X_test_scaled)

X_train_reduced_np = np.array(X_train_reduced)
X_test_reduced_np = np.array(X_test_reduced)

print("Number of original features is {} and of reduced features is {}".format(X_train.shape[1], X_train_reduced.shape[1]))
print("Number of original features is {} and of reduced features is {}".format(X_test.shape[1], X_test_reduced.shape[1]))

kernel_evals = {}

def evaluate_classification_XGBRF_cuML_SVC(rf_model, svc_model, name,
                                           X_train, X_test, y_train, y_test):
    # Transform features using RF probabilities (on GPU)
    rf_train_features = rf_model.predict_proba(X_train)
    rf_test_features = rf_model.predict_proba(X_test)

    # SVC prediction (still on GPU)
    y_train_pred = svc_model.predict(rf_train_features)
    y_test_pred = svc_model.predict(rf_test_features)

    # Convert to CPU NumPy for sklearn metrics
    y_train_pred = cp.asnumpy(y_train_pred)
    y_test_pred = cp.asnumpy(y_test_pred)
    y_train_cpu = cp.asnumpy(y_train)
    y_test_cpu = cp.asnumpy(y_test)

    # Multi-class metrics
    train_precision = metrics.precision_score(y_train_cpu, y_train_pred, average='weighted')
    test_precision  = metrics.precision_score(y_test_cpu, y_test_pred, average='weighted')

    train_accuracy = metrics.accuracy_score(y_train_cpu, y_train_pred)
    test_accuracy  = metrics.accuracy_score(y_test_cpu, y_test_pred)

    train_recall = metrics.recall_score(y_train_cpu, y_train_pred, average='weighted')
    test_recall  = metrics.recall_score(y_test_cpu, y_test_pred, average='weighted')

    train_f1 = metrics.f1_score(y_train_cpu, y_train_pred, average='weighted')
    test_f1  = metrics.f1_score(y_test_cpu, y_test_pred, average='weighted')

    # Store and print results
    kernel_evals[str(name)] = [train_accuracy, test_accuracy,
                               train_precision, test_precision,
                               train_recall, test_recall]
    print(f"\nTraining Accuracy {name} {train_accuracy*100:.2f}%  Test Accuracy {name} {test_accuracy*100:.2f}%")
    print(f"Training Precision {name} {train_precision*100:.2f}%  Test Precision {name} {test_precision*100:.2f}%")
    print(f"Training Recall {name} {train_recall*100:.2f}%  Test Recall {name} {test_recall*100:.2f}%")
    print(f"Training F1 Score {name} {train_f1*100:.2f}%  Test F1 Score {name} {test_f1*100:.2f}%")


# ====== Convert your data to GPU before training ======
X_train_gpu = cp.asarray(X_train_reduced)   # or cudf.DataFrame.from_pandas(X_train_reduced)
X_test_gpu  = cp.asarray(X_test_reduced)
y_train_gpu = cp.asarray(y_train.values)
y_test_gpu  = cp.asarray(y_test.values)

# Train XGBRF on GPU
XGB_RF = xgb.XGBRFClassifier(
    objective='multi:softprob',  # <-- multi-class
    num_class=len(cp.unique(y_train_gpu)),
    n_estimators=100,
    random_state=42,
    tree_method='hist',
    device='cuda'
)
XGB_RF.fit(X_train_gpu, y_train_gpu)

# Prepare features for cuML SVC (GPU-based RF probabilities)
RF_train_features = XGB_RF.predict_proba(X_train_gpu)
RF_test_features  = XGB_RF.predict_proba(X_test_gpu)

# Train cuML SVM (on GPU)
SVM_classifier = cuSVC(kernel='rbf', probability=True)
SVM_classifier.fit(RF_train_features, y_train_gpu)

# Evaluate hybrid (no CPU-GPU mismatch warning now)
evaluate_classification_XGBRF_cuML_SVC(
    XGB_RF, SVM_classifier,"Hybrid XGBRF + cuML SVM",
    X_train_gpu, X_test_gpu, y_train_gpu, y_test_gpu
)