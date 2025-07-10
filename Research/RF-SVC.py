import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    All features are numeric.
"""
X = df.drop('label', axis=1)
y = df['label']
#print(X.head())
#print(y.head())

X2 = df.drop('label', axis=1)
y2 = df['label']


X_train, X_test, y_train, y_test = TTS(X, y, test_size=0.2, random_state=42)

# Targets are not numeric
LE = LabelEncoder()
y_train_encoded = LE.fit_transform(y_train)
y_test_encoded = LE.transform(y_test)

# Statistical summary
print('Spoofing dataset:\n')
print('Total samples:', len(df))
print('Training samples:', len(X_train))
print('Testing samples:', len(X_test))
print('Number of features:', X.shape[1])
print('\nFeature Statistics:\n', X.describe())
print('\nClass distribution (full datasets):\n', y.value_counts())
print('Number of targets:', y.shape[0])

print('\nJamming dataset:\n')
print('Total samples:', len(df2))
print('Number of features:', X2.shape[1])
print('\nFeature Statistics:\n', X2.describe())
print('\nClass distribution (full datasets):\n', y2.value_counts())
print('Number of targets:', y2.shape[0])