import pandas as pd

from io import StringIO
from collections import defaultdict
from DataPreprocess import DataPreprocess
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC as SVC

def load_and_split_by_class(filename):
    dfs_by_class = defaultdict(list)

    with open(filename, 'r') as f:
        lines = f.readlines()

    header_line = None
    chunk_lines = []

    for line in lines:
        if line.startswith('timestamp_c'):
            # Process previous chunk if exists
            if chunk_lines:
                chunk_csv = ''.join(chunk_lines)
                chunk_df = pd.read_csv(StringIO(chunk_csv))

                # --- START FIXES ---
                # 1. Drop columns with all NaNs (empty trailing columns)
                chunk_df = chunk_df.dropna(axis=1, how='all')

                # 2. Rename 'class' to 'Label' if needed (adjust according to your pipeline)
                if 'class' in chunk_df.columns:
                    chunk_df.rename(columns={'class': 'Label'}, inplace=True)

                # 3. Drop columns with >50% NaNs (optional threshold)
                chunk_df = chunk_df.loc[:, chunk_df.isna().mean() < 0.5]

                # 4. Fill remaining NaNs with 0 (or use another imputation)
                chunk_df = chunk_df.fillna(0)
                # --- END FIXES ---

                # Group by class/label and store
                for class_label, group_df in chunk_df.groupby(chunk_df['Label'].str.lower()):
                    dfs_by_class[class_label].append(group_df)

                chunk_lines = []
            header_line = line
        chunk_lines.append(line)

    # Process the last chunk (same fixes applied)
    if chunk_lines:
        chunk_csv = ''.join(chunk_lines)
        chunk_df = pd.read_csv(StringIO(chunk_csv))

        # --- START FIXES ---
        chunk_df = chunk_df.dropna(axis=1, how='all')
        if 'class' in chunk_df.columns:
            chunk_df.rename(columns={'class': 'Label'}, inplace=True)
        chunk_df = chunk_df.loc[:, chunk_df.isna().mean() < 0.5]
        chunk_df = chunk_df.fillna(0)
        # --- END FIXES ---

        for class_label, group_df in chunk_df.groupby(chunk_df['Label'].str.lower()):
            dfs_by_class[class_label].append(group_df)

    # Concatenate chunks per class
    concatenated = {cls: pd.concat(dfs, ignore_index=True) for cls, dfs in dfs_by_class.items()}
    return concatenated

# Usage remains the same
filename = 'Dataset_T-ITS.csv'
dfs_by_class = load_and_split_by_class(filename)

benign_df = dfs_by_class.get('benign')
dos_df = dfs_by_class.get('dos attack')
replay_df = dfs_by_class.get('replay')
evil_twin_df = dfs_by_class.get('evil_twin')
fdi_df = dfs_by_class.get('fdi')

print(f"[CYBER DATAFRAME INFO] ALL BELOW IS CYBER DATA AND NOT PHYSICAL DATA")
print(f"[DATAFRAME INFO] Benign shape: {benign_df.shape if benign_df is not None else 'No data'}")
print(f"[DATAFRAME INFO] DoS shape: {dos_df.shape if dos_df is not None else 'No data'}")
print(f"[DATAFRAME INFO] Replay shape: {replay_df.shape if replay_df is not None else 'No data'}")
print(f"[DATAFRAME INFO] Evil Twin shape: {evil_twin_df.shape if evil_twin_df is not None else 'No data'}")
print(f"[DATAFRAME INFO] FDI shape: {fdi_df.shape if fdi_df is not None else 'No data'}")

print(f"[DATA INFO] Total data: {benign_df.shape[0] + dos_df.shape[0] + replay_df.shape[0] + evil_twin_df.shape[0] + fdi_df.shape[0] + 5}")
print(f"[DATA INFO] Benign data amount: {benign_df.shape[0] + 1}")
print(f"[DATA INFO] All attacks amount: {dos_df.shape[0] + replay_df.shape[0] + evil_twin_df.shape[0] + fdi_df.shape[0] + 4}")

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

X = final_df.drop(columns='Label')
y = final_df['Label']
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
