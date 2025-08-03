import pandas as pd
import glob
import xgboost as xgb
from cuml.svm import SVC as cuSVC
import numpy as np
import cupy as cp

from DataPreprocess import DataPreprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics

"""
NOTE:

Switching machine learning libraries for GPU support as the dataset is large.
    - XGBoost Random Forest.
    - ThunderSVM for SVC.

For now, smaller datasets will be used with
    - Scikit-learn RF and SVC (CPU-based).

For larger datasets, switch to:
    - XGB-RF + T-SVC (GPU-accelerated with XGBoost and ThunderSVM).

Ensure to install the required libraries:
    - pip install xgboost
    - pip install thundersvm
"""

bin_files = glob.glob("UAVCAM-Attack/*.bin")

column_names = [
    'Status', 'Time', 'Interface', 'ID', 'Length', 'Data1', 'Data2', 
    'Data3', 'Data4', 'Data5', 'Data6', 'Data7','Data8'
]

df = pd.concat((pd.read_csv(
    file, sep=r'\s+', names=column_names) for file in bin_files if file.endswith('.bin')))

print("Combined shape:", df.shape)
print(df.head())

X = df.drop(columns=['Status'])
y = df['Status']

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Class distribution in y:", y.value_counts())
print('Unique classes in y:', y.unique())

DP = DataPreprocess()
df = DP.runNew(targetName='Status', featuresToBinary=True, targetToBinary=True, givenDF=df, normalBehavior='Normal')
print("Cleaned shape:", df.shape)
print(df.head())


X = df.drop(columns=['Status'])
y = df['Status']

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Class distribution in y:", y.value_counts())
print('Unique classes in y:', y.unique())

corr_with_target = df.corr()['Status'].drop('Status').abs()
leaking_features = corr_with_target[corr_with_target > 0.9].index.tolist()

if leaking_features:
    print(f"[LEAKAGE WARNING] Dropping features correlated with target > 0.9: {leaking_features}")
else:
    print("[LEAKAGE CHECK] No data leakage detected.")

print("Class distribution in y:", y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("y_train distribution:", y_train.value_counts())
print("y_test distribution:", y_test.value_counts())

print('Train shape:', X_train.shape)
print('Test shape:', X_test.shape)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=0.95)
X_train_reduced = pca.fit_transform(X_train_scaled)
X_test_reduced = pca.transform(X_test_scaled)

print("Number of original features is {} and of reduced features is {}".format(X_train.shape[1], X_train_reduced.shape[1]))
print("Number of original features is {} and of reduced features is {}".format(X_test.shape[1], X_test_reduced.shape[1]))

X_train_reduced_np = np.array(X_train_reduced)
X_test_reduced_np = np.array(X_test_reduced)

# Convert NumPy arrays to CuPy arrays
X_train_gpu = cp.asarray(X_train_reduced_np)
X_test_gpu = cp.asarray(X_test_reduced_np)

XGB_RF = xgb.XGBRFClassifier(
    objective='binary:logistic',
    n_estimators=100,
    random_state=42,
    tree_method='hist', 
    device='cuda'
)
XGB_RF.fit(X_train_reduced_np, y_train)

# cuML SVC (GPU)
SVM_classifier = cuSVC(kernel='rbf', probability=True)
SVM_classifier.fit(X_train_reduced_np, y_train.values)  # cuML expects ndarray for labels

def evaluate_classification_cuml(model, name, X_train, X_test, y_train, y_test):
    # cuML's predict returns cupy array; convert to numpy for sklearn metrics
    y_train_pred = model.predict(X_train).get() if hasattr(model.predict(X_train), 'get') else model.predict(X_train)
    y_test_pred = model.predict(X_test).get() if hasattr(model.predict(X_test), 'get') else model.predict(X_test)

    train_precision = metrics.precision_score(y_train, y_train_pred)
    test_precision = metrics.precision_score(y_test, y_test_pred)

    train_accuracy = metrics.accuracy_score(y_train, y_train_pred)
    test_accuracy = metrics.accuracy_score(y_test, y_test_pred)

    train_recall = metrics.recall_score(y_train, y_train_pred)
    test_recall = metrics.recall_score(y_test, y_test_pred)

    train_f1 = metrics.f1_score(y_train, y_train_pred)
    test_f1 = metrics.f1_score(y_test, y_test_pred)

    cm_train = metrics.confusion_matrix(y_train, y_train_pred)
    tp_train = cm_train[1, 1]
    fn_train = cm_train[1, 0]
    idc_train = tp_train / (tp_train + fn_train) if (tp_train + fn_train) > 0 else 0.0

    cm_test = metrics.confusion_matrix(y_test, y_test_pred)
    tp_test = cm_test[1, 1]
    fn_test = cm_test[1, 0]
    idc_test = tp_test / (tp_test + fn_test) if (tp_test + fn_test) > 0 else 0.0

    print(f"\nTraining Accuracy {name} {train_accuracy*100:.2f}%  Test Accuracy {name} {test_accuracy*100:.2f}%")
    print(f"Training Precision {name} {train_precision*100:.2f}%  Test Precision {name} {test_precision*100:.2f}%")
    print(f"Training Recall {name} {train_recall*100:.2f}%  Test Recall {name} {test_recall*100:.2f}%")
    print(f"Training F1 Score {name} {train_f1*100:.2f}%  Test F1 Score {name} {test_f1*100:.2f}%")
    print(f"Training IDC {name} {idc_train*100:.2f}%  Test IDC {name} {idc_test*100:.2f}%")

kernal_evals = {}
def evaluate_classification_XGB_RF(model, name, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_precision = metrics.precision_score(y_train, y_train_pred)
    test_precision = metrics.precision_score(y_test, y_test_pred)

    train_accuracy = metrics.accuracy_score(y_train, y_train_pred)
    test_accuracy = metrics.accuracy_score(y_test, y_test_pred)

    train_recall = metrics.recall_score(y_train, y_train_pred)
    test_recall = metrics.recall_score(y_test, y_test_pred)

    train_f1 = metrics.f1_score(y_train, y_train_pred)
    test_f1 = metrics.f1_score(y_test, y_test_pred)

    cm_train = metrics.confusion_matrix(y_train, y_train_pred)
    tp_train = cm_train[1, 1]
    fn_train = cm_train[1, 0]
    idc_train = tp_train / (tp_train + fn_train) if (tp_train + fn_train) > 0 else 0.0

    cm_test = metrics.confusion_matrix(y_test, y_test_pred)
    tp_test = cm_test[1, 1]
    fn_test = cm_test[1, 0]
    idc_test = tp_test / (tp_test + fn_test) if (tp_test + fn_test) > 0 else 0.0

    kernal_evals[str(name)] = [train_accuracy, test_accuracy, train_precision, test_precision, train_recall, test_recall]
    print(f"\nTraining Accuracy {name} {train_accuracy*100:.2f}%  Test Accuracy {name} {test_accuracy*100:.2f}%")
    print(f"Training Precision {name} {train_precision*100:.2f}%  Test Precision {name} {test_precision*100:.2f}%")
    print(f"Training Recall {name} {train_recall*100:.2f}%  Test Recall {name} {test_recall*100:.2f}%")
    print(f"Training F1 Score {name} {train_f1*100:.2f}%  Test F1 Score {name} {test_f1*100:.2f}%")
    print(f"Training IDC {name} {idc_train*100:.2f}%  Test IDC {name} {idc_test*100:.2f}%")

# Evaluate both
evaluate_classification_XGB_RF(XGB_RF, "Random Forest", X_train_reduced_np, X_test_reduced_np, y_train, y_test)
evaluate_classification_cuml(SVM_classifier, "cuML SVM", X_train_reduced_np, X_test_reduced_np, y_train, y_test)