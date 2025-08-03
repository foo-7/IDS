import pandas as pd

from DataPreprocess import DataPreprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC as SVC

fileName1 = 'Dataset_T-ITS.csv'

DP = DataPreprocess(fileName=fileName1)
df = DP.runNew(targetName='class', featuresToBinary=True, targetToBinary=True)

X = df.drop(columns=['class'])
y = df['class']

print("X shape:", X.shape)
print("y shape:", y.shape)

corr_with_target = df.corr()['class'].drop('class').abs()
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
def evaluate_classification(model, name, X_train, X_test, y_train, y_test):
    train_precision = metrics.precision_score(y_train, model.predict(X_train))
    test_precision = metrics.precision_score(y_test, model.predict(X_test))

    train_accuracy = metrics.accuracy_score(y_train, model.predict(X_train))
    test_accuracy = metrics.accuracy_score(y_test, model.predict(X_test))

    train_recall = metrics.recall_score(y_train, model.predict(X_train))
    test_recall = metrics.recall_score(y_test, model.predict(X_test))

    train_f1 = metrics.f1_score(y_train, model.predict(X_train))
    test_f1 = metrics.f1_score(y_test, model.predict(X_test))

    cm_train = metrics.confusion_matrix(y_train, model.predict(X_train))
    tp_train = cm_train[1, 1]
    fn_train = cm_train[1, 0]
    idc_train = tp_train / (tp_train + fn_train) if (tp_train + fn_train) > 0 else 0.0

    cm_test = metrics.confusion_matrix(y_test, model.predict(X_test))
    tp_test = cm_test[1, 1]
    fn_test = cm_test[1, 0]
    idc_test = tp_test / (tp_test + fn_test) if (tp_test + fn_test) > 0 else 0.0

    kernal_evals[str(name)] = [train_accuracy, test_accuracy, train_precision, test_precision, train_recall, test_recall]
    print("\nTraining Accuracy " + str(name) + " {}  Test Accuracy ".format(train_accuracy*100) + str(name) + " {}".format(test_accuracy*100))
    print("Training Precesion " + str(name) + " {}  Test Precesion ".format(train_precision*100) + str(name) + " {}".format(test_precision*100))
    print("Training Recall " + str(name) + " {}  Test Recall ".format(train_recall*100) + str(name) + " {}".format(test_recall*100))
    print("Training F1 Score " + str(name) + " {}  Test F1 Score ".format(train_f1*100) + str(name) + " {}".format(test_f1*100))
    print("Training IDC " + str(name) + " {}  Test IDC ".format(idc_train * 100) + str(name) + " {}".format(idc_test * 100))

RF_classifier = RFC(n_estimators=100, random_state=42)
RF_classifier.fit(X_train_reduced, y_train)
SVM_classifier = SVC()
SVM_classifier.fit(X_train_reduced, y_train)
evaluate_classification(RF_classifier, "Random Forest", 
                        X_train_reduced, X_test_reduced,
                        y_train, y_test)

evaluate_classification(SVM_classifier, "SVM",
                        X_train_reduced, X_test_reduced,
                        y_train, y_test)