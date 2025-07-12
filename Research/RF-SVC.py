import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Given ML models: Random Forest, Support Vector Classifier
# These will classify if there is an intrusion within the system
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC as SVC

# Data preprocessing
# LabelEncoder   -> We will need to convert the feature 'label" into a numeric
# MinMaxScaler   -> We will need to normalize numeric features into a fix range [0,1] (ONLY USE ON NN)
# StandardScaler -> We will need to normalize numeric features into a fix range 
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# Maybe consider this?
# Module that helps you chain together multiple preprocessing steps and a model into a single, clean workflow.

from sklearn.model_selection import train_test_split as TTS

df = pd.read_csv('spoofing-merged-gps-only.csv')
df2 = pd.read_csv('jamming-merged-gps-only.csv')
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
print(df.shape)
#print(df_removed.shape)

"""
    Remove outliers
"""

quantitative_data = df_removed.select_dtypes(include='number').copy()
target = quantitative_data['label']
features = quantitative_data.drop(columns=['label'])
#print('Quantitative data info:\n', quantitative_data.info())

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
print(df.shape)

"""
    Splitting data
"""
X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = TTS(X, y, test_size=0.2, random_state=42)

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
pca = pca.fit(X_train)
X_train_reduced = pca.transform(X_train)

pca = pca.fit(X_test)
X_test_reduced = pca.transform(X_test)

print("Number of original features is {} and of reduced features is {}".format(X_train.shape[1], X_train_reduced.shape[1]))
print("Number of original features is {} and of reduced features is {}".format(X_test.shape[1], X_test_reduced.shape[1]))

"""
    Evaluation and Deployment
"""

