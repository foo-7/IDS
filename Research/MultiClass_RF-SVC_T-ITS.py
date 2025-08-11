import pandas as pd

from DataPreprocess import DataPreprocess
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC as SVC


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

print(f'[DATAFRAME INFO] X_train size: {X_train.shape}')
print(f'[DATAFRAME INFO] X_test size: {X_test.shape}')

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