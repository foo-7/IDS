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
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
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
#print(df.shape)
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
print(df_filtered.shape)

"""
    Usage of PCA
"""


"""
    Check if any binary variables
"""
for col in df_removed.columns:
    unique_vals = df[col].dropna().unique()
    if len(unique_vals) == 2:
        print(f"Binary variable candidate: '{col}' --> {unique_vals}")
"""
    Our targets are binary: [benign, malicious]

    Our features are all numeric.
"""

"""
    All features are numeric.
"""
X = df.drop('label', axis=1)
y = df['label']

# Maybe we should do this in the beginning????
X_train, X_test, y_train, y_test = TTS(X, y, test_size=0.2, random_state=42)


# X2 = df.drop('label', axis=1)
# y2 = df['label']


# X_train, X_test, y_train, y_test = TTS(X, y, test_size=0.2, random_state=42)

# # Targets are not numeric
# LE = LabelEncoder()
# y_train_encoded = LE.fit_transform(y_train)
# y_test_encoded = LE.transform(y_test)

# # For Professor to look at statistical summary
# # Statistical summary
# print('Spoofing dataset:\n')
# print('Total samples:', len(df))
# print('Training samples:', len(X_train))
# print('Testing samples:', len(X_test))
# print('Number of features:', X.shape[1])
# print('\nFeature Statistics:\n', X.describe())
# print('\nClass distribution (full datasets):\n', y.value_counts())
# print('Number of targets:', y.shape[0])

# print('\nJamming dataset:\n')
# print('Total samples:', len(df2))
# print('Number of features:', X2.shape[1])
# print('\nFeature Statistics:\n', X2.describe())
# print('\nClass distribution (full datasets):\n', y2.value_counts())
# print('Number of targets:', y2.shape[0])

# Data preprocessing
