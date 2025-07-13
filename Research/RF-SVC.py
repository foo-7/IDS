import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Given ML models: Random Forest, Support Vector Classifier
# These will classify if there is an intrusion within the system
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC as SVC

# Data preprocessing
# MinMaxScaler   -> We will need to normalize numeric features into a fix range [0,1] (ONLY USE ON NN)
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# Maybe consider this?
# Module that helps you chain together multiple preprocessing steps and a model into a single, clean workflow.
# from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split as TTS

# To evaluate
from sklearn import metrics

# Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV, cross_val_score

df = pd.read_csv('spoofing-merged-gps-only.csv')
#df = pd.read_csv('jamming-merged-gps-only.csv')
#print(df.head())
"""
    There is no NaN values
    There is no duplicated values
"""
#print(df.isnull().values.any())
#print(df.duplicated().values.any())
print(df.shape)

# Just in case if there is any duplicates or NaN values
if df.isnull().values.any():

    df = df.dropna(axis=1)

if df.duplicated().values.any():
    df.drop_duplicates(keep='first', inplace=True)
    df.reset_index(drop=True, inplace=True)

# We change the targets from categorical to numeric
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
df['label'] = df['label'].replace({
    'benign' : 0,
    'malicious' : 1
})

"""
    Dropping columns with high correlation

    Dropping highly correlated columns removes redundancy
    Helps the model learn more distinct and meaningful relationships
    Reduces overfitting caused by redundant features
    Simplifies the model and often improves performance

    Why low correlation?

    Feature Removal usually focuses on highly redundant (highly correlated) features
    or features that don't improve model performance in other ways.
"""
corr_df = df.corr().abs()
mask = np.triu(np.ones_like(corr_df, dtype=bool))
tri_df = corr_df.mask(mask)

to_drop = [c for c in tri_df.columns if any(tri_df[c] >= 0.9)]
#print(to_drop)

df_removed = df.drop(columns=to_drop, axis=1)
#print(df_removed.shape)
#print(df_removed['label'].nunique())

"""
    Remove outliers

    CAUTION: IT REMOVES ALL THE MALICIOUS TARGETS
"""

quantitative_data = df_removed.select_dtypes(include='number').copy()
target = quantitative_data['label']
features = quantitative_data.drop(columns=['label'])
#print('Quantitative data info:\n', quantitative_data.info())

""" MIGHT REMOVE SINCE IT REMOVES ALL THE MALICIOUS LABELS IN THE DATASET
    BELOW WILL NOT REMOVE THE MALICIOUS
def remove_outliers(data, k=1.5):
    outliers = pd.DataFrame()
    for column in data.columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        LB = Q1 - k * IQR
        UB = Q3 + k * IQR

        column_outliers = data[(data[column] < LB) | (data[column] > UB)]
        outliers = pd.concat([outliers, column_outliers])

    return outliers

outlier = remove_outliers(features)
# ~ NOT operation
feature_filtered_data = features[~features.index.isin(outlier.index)]
#print(feature_filtered_data.shape)
target_filtered_data = target[~target.index.isin(outlier.index)]
#print(target_filtered_data.shape)
df_filtered = feature_filtered_data.loc[:, (feature_filtered_data != 0).any(axis=0)].copy()
df_filtered.loc[:, 'label'] = target_filtered_data
df = pd.DataFrame(df_filtered)
#print(df.shape)
"""

def remove_outliers_per_class(features, target, k=1.5):
    indices_to_keep = []
    for label in target.unique():
        class_data = features[target == label]
        outlier_indices = set()

        for col in class_data.columns:
            Q1 = class_data[col].quantile(0.25)
            Q3 = class_data[col].quantile(0.75)
            IQR = Q3 - Q1
            LB = Q1 - k * IQR
            UB = Q3 + k * IQR

            outliers_col = class_data[(class_data[col] < LB) | (class_data[col] > UB)].index
            outlier_indices.update(outliers_col)

        # Keep samples that are NOT outliers for this class
        keep_indices = set(class_data.index) - outlier_indices
        indices_to_keep.extend(keep_indices)

    return features.loc[indices_to_keep], target.loc[indices_to_keep]

filtered_features, filtered_target = remove_outliers_per_class(features, target, k=1.5)

df_filtered = filtered_features.copy()
df_filtered['label'] = filtered_target

"""
    Splitting data
"""
# Outlier removed
#X = df.drop('label', axis=1)
#y = df['label']

X = filtered_features
y = filtered_target

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = TTS(X, y, test_size=0.2, random_state=42, stratify=y)

print(y_train.value_counts())
print(y_test.value_counts())


"""
    Data normalization

    Scaling/transforming targets only makes sense if you are doing regression and the
    target is continuous

    Since we are dealing with binary classification:
        0 for benign,
        1 for malicious
    We do not scale them.
"""
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# scalar returned a NumPy array to df_scaled, so we need to make it in a DataFrame
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)


"""
    PCA
"""
pca = PCA(n_components=0.95)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)

print(X_train_reduced.shape)
print(X_test_reduced.shape)

print("Number of original features is {} and of reduced features is {}".format(X_train.shape[1], X_train_reduced.shape[1]))
print("Number of original features is {} and of reduced features is {}".format(X_test.shape[1], X_test_reduced.shape[1]))

"""
    Evaluation and Deployment
"""
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
    print("Training Accuracy " + str(name) + " {}  Test Accuracy ".format(train_accuracy*100) + str(name) + " {}".format(test_accuracy*100))
    print("Training Precesion " + str(name) + " {}  Test Precesion ".format(train_precision*100) + str(name) + " {}".format(test_precision*100))
    print("Training Recall " + str(name) + " {}  Test Recall ".format(train_recall*100) + str(name) + " {}".format(test_recall*100))
    print("Training F1 Score " + str(name) + " {}  Test F1 Score ".format(train_f1*100) + str(name) + " {}".format(test_f1*100))
    print("Training IDC " + str(name) + " {}  Test IDC ".format(idc_train * 100) + str(name) + " {}".format(idc_test * 100))

    actual = y_test
    predicted = model.predict(X_test)
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['normal', 'attack'])

    fig, ax = plt.subplots(figsize=(10,10))
    ax.grid(False)
    cm_display.plot(ax=ax)
    plt.show()

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

rf_importances = RF_classifier.feature_importances_
top_components_idx = np.argsort(rf_importances)[::-1]  # descending order
n_components_to_check = 3
n_top_features_per_component = 5

for comp_idx in top_components_idx[:n_components_to_check]:
    print(f"\nTop original features for PCA Component {comp_idx} "
          f"(RF importance = {rf_importances[comp_idx]:.4f}):")
    
    component = pca.components_[comp_idx]
    top_feature_indices = np.argsort(np.abs(component))[-n_top_features_per_component:]
    
    for i in reversed(top_feature_indices):
        print(f"  {X_train.columns[i]}: {component[i]:.4f}")


"""
    Since we want to use both models,
        is it great to consider the following:
        1. Stacking Ensemble
        2. Voting Ensemble

        - Provided by ChatGPT (need to ask professor)
"""

"""
    Hyperparameter Tuning
"""
