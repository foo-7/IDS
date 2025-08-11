import pandas as pd
# Python module used to find all file apths that match a specified pattern.
# Used to read multiple files found in UAVCAM-Attack dataset folder.
import glob
import numpy as np
import cupy as cp
import xgboost as xgb
from cuml.svm import SVC as cuSVC

from DataPreprocess import DataPreprocess
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics

bin_files = glob.glob('UAVCAN-Attack/*.bin')
column_names = [
    'Status', 'Time', 'Interface', 'ID', 'Length', 'Data1', 'Data2', 
    'Data3', 'Data4', 'Data5', 'Data6', 'Data7','Data8'
]

df = pd.concat(pd.read_csv(
    file, sep=r'\s+', names=column_names) for file in bin_files if file.endswith('.bin'
))

print(f'[DATAFRAME INFO] Dataframe shape: {df.shape}')
#print(f'[INFO] The amount of unique label values found: {len(df['Status'].unique())}')
#print(f'[INFO] Display of all unique labels:\n{df["Status"].unique()}')

DP = DataPreprocess()
df = DP.runNew(
    targetName='Status',
    featuresToBinary= True,
    targetToBinary= True if len(df['Status'].unique()) == 2 else False,
    givenDF= df,
    normalBehavior='Normal'
)

df = shuffle(df, random_state=42)
X = df.drop(columns='Status')
y = df['Status']

corr_with_target = df.corr()['Status'].drop('Status').abs()
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