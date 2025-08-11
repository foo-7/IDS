import pandas as pd

from DataPreprocess import DataPreprocess
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC as SVC

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

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=0.95)
X_train_reduced = pca.fit_transform(X_train_scaled)
X_test_reduced = pca.transform(X_test_scaled)

print("Number of original features is {} and of reduced features is {}".format(X_train.shape[1], X_train_reduced.shape[1]))
print("Number of original features is {} and of reduced features is {}".format(X_test.shape[1], X_test_reduced.shape[1]))

kernal_evals = {}
def evaluate_classification_hybrid(rf_model, svc_model, name, X_train, X_test, y_train, y_test):
    rf_train_features = rf_model.predict_proba(X_train)
    rf_test_features = rf_model.predict_proba(X_test)

    y_train_pred = svc_model.predict(rf_train_features)
    y_test_pred = svc_model.predict(rf_test_features)

    # Multi-class metrics
    train_precision = metrics.precision_score(y_train, y_train_pred, average='weighted')
    test_precision = metrics.precision_score(y_test, y_test_pred, average='weighted')

    train_accuracy = metrics.accuracy_score(y_train, y_train_pred)
    test_accuracy = metrics.accuracy_score(y_test, y_test_pred)

    train_recall = metrics.recall_score(y_train, y_train_pred, average='weighted')
    test_recall = metrics.recall_score(y_test, y_test_pred, average='weighted')

    train_f1 = metrics.f1_score(y_train, y_train_pred, average='weighted')
    test_f1 = metrics.f1_score(y_test, y_test_pred, average='weighted')

    print(metrics.classification_report(y_test, y_test_pred))

    kernal_evals[str(name)] = [train_accuracy, test_accuracy, train_precision, test_precision, train_recall, test_recall]
    print(f"\nTraining Accuracy {name}: {train_accuracy*100:.5f}%  Test Accuracy {name}: {test_accuracy*100:.5f}%")
    print(f"Training Precision {name}: {train_precision*100:.5f}%  Test Precision {name}: {test_precision*100:.5f}%")
    print(f"Training Recall {name}: {train_recall*100:.5f}%  Test Recall {name}: {test_recall*100:.5f}%")
    print(f"Training F1 Score {name}: {train_f1*100:.5f}%  Test F1 Score {name}: {test_f1*100:.5f}%")

RF_classifier = RFC(n_estimators=100, random_state=42)
RF_classifier.fit(X_train_reduced, y_train)

RF_train_features = RF_classifier.predict_proba(X_train_reduced)

SVM_classifier = SVC(probability=True, random_state=42)
SVM_classifier.fit(RF_train_features, y_train)

evaluate_classification_hybrid(
    RF_classifier, SVM_classifier, "Hybrid RF-SVC",
    X_train_reduced, X_test_reduced,
    y_train, y_test
)