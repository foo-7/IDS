from DataPreprocess import DataPreprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC as SVC

fileName1 = "jamming-merged-gps-only.csv"
fileName2 = "spoofing-merged-gps-only.csv"

DP = DataPreprocess(fileName=fileName2)
df = DP.runNew(targetName='label', featuresToBinary=True, targetToBinary=True)

X = df.drop(columns=['label'])
y = df['label']

print("X shape:", X.shape)
print("y shape:", y.shape)

corr_with_target = df.corr()['label'].drop('label').abs()
leaking_features = corr_with_target[corr_with_target > 0.9].index.tolist()

if leaking_features:
    print(f"[LEAKAGE WARNING] Dropping features correlated with target > 0.9: {leaking_features}")
else:
    print("[LEAKAGE CHECK] No data leakage detected.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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

kernal_evals = {}
def evaluate_classification_hybrid(rf_model, svc_model, name, X_train, X_test, y_train, y_test):
    # Transform features using RF probabilities
    rf_train_features = rf_model.predict_proba(X_train)
    rf_test_features = rf_model.predict_proba(X_test)

    # Predictions from SVC using RF-transformed features
    y_train_pred = svc_model.predict(rf_train_features)
    y_test_pred = svc_model.predict(rf_test_features)

    # Calculate metrics
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

    # Store and print results
    kernal_evals[str(name)] = [train_accuracy, test_accuracy, train_precision, test_precision, train_recall, test_recall]
    print(f"\nTraining Accuracy {name} {train_accuracy*100}  Test Accuracy {name} {test_accuracy*100}")
    print(f"Training Precesion {name} {train_precision*100}  Test Precesion {name} {test_precision*100}")
    print(f"Training Recall {name} {train_recall*100}  Test Recall {name} {test_recall*100}")
    print(f"Training F1 Score {name} {train_f1*100}  Test F1 Score {name} {test_f1*100}")
    print(f"Training IDC {name} {idc_train*100}  Test IDC {name} {idc_test*100}")

RF_classifier = RFC(n_estimators=100, random_state=42)
RF_classifier.fit(X_train_reduced, y_train)

# We use RF to transform features into probabilities for SVC
RF_train_features = RF_classifier.predict_proba(X_train_reduced)

SVM_classifier = SVC(probability=True, random_state=42)
SVM_classifier.fit(RF_train_features, y_train)

evaluate_classification_hybrid(RF_classifier, SVM_classifier, "Hybrid RF-SVC",
                               X_train_reduced, X_test_reduced,
                               y_train, y_test)