"""
    NSL-KDD Dataset is a refined version of the KDD cup 99 dataset. It contains
    essential records of it predecessor balancing the proportions of normal versus
    attack traces, and excluding redundant records.
        -> Each record is composed of 41 attributes unfollding four different types
           of features of the flow, and its assigned label which classifies it as an
           attack or as normal.
        -> These features include:
            a. basic characteristics of each network connection vector
                = duration or the number of bytes transferred
                = content related features:
                    a. number of connections to the same destination
                = host-based traffic features:
                    a. number of connections to the same port number.

    The whole amount of records covers one normal class and four attack classes:
        a. Denial of Service (DoS)
            - Restrict the user from using certain services.
            - Overloading the system or keep the resources busy in the network.

        b. Surveillance (Probe)
            - Attacker tries to gain access to all data of the system.
            - Have full control on the server. 

        c. Unauthorized access to local super user (R2L)
            - Gain access to a system by sending some message to the server
              and gaining access to the system from a remote machine.
            - Changes to the server to get access to resources.
                = One example is the "guess password" attack.

        d. Unauthorized access from a remote machine (U2R)
            - Aims to analysing the network, gather information,
            - Performed to be able to attack through some other methods later.
"""

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Powerful data visualization library built on top of matplotlib
import seaborn as sns

# Encoder used to convert categorial variables into a format that can be provided
# to ML algorithms to improve predictions.
from sklearn.preprocessing import OneHotEncoder

# Scales (normalizes) numeric feature to a fixed range, usually [0, 1].
from sklearn.preprocessing import MinMaxScaler

# Converts categorial labels (strings or non-numeric values) into ints.
from sklearn.preprocessing import LabelEncoder

# Random forest uses multiple decision trees to improve accuracy and reduce overfitting.
# ML MODEL (classification, supervised) -> balanced performance and interpretability, data non-linearity.
from sklearn.ensemble import RandomForestClassifier

# VarianceThreshold is a feature selector that removes all features whose variance that does not meet the threshold.
# SelectFromModel selects features based on import socres from a model.
from sklearn.feature_selection import SelectFromModel, VarianceThreshold

# SelectKBest selects features according to the k highest scores.
# Chi2 is used with SelectKBest to measure how much each feature is related to the target.
from sklearn.feature_selection import SelectKBest, chi2

# Support Vector Classification (SVC) used for finding the best hyperplane that separates classes in
# the feature space with the maximum margin.
# ML MODEL (classification, supervised) -> bi/multi-class classification, text classification, image classification, high accuracy
from sklearn.svm import SVC

# Accuracy score -> measures correct predictions (overall accuracy).
# Classification report -> precision, recall, f1-score, support for each class.
# Confusion matrix -> matrix of true versus predicted labels (true positives, true negatives, false positives, false negatives)
# ROC AUC score -> area under the receiver operating characteristic curve (true positive rate vs false positive rate)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# GM assumes data is generated from a mixture of several Gaussian distributions.
# Tries to find the parameters (mean, covariance) that best fit the data.
# Probabilistic model that can be used for clustering, which assigns probabilities of each point belonging to a cluster.
# ML MODEL (clustering, unsupervised) -> anomaly detection, density estimation, clustering data
from sklearn.mixture import GaussianMixture
from sklearn import metrics

# Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms data into a lower-dimensional space.
# Reduces the number of features while retaining most of the information (reduction of redundant features).
from sklearn.decomposition import PCA

# Visualization tracking progress of long-running tasks.
from tqdm import tqdm

# Popular unsupervised ML model for clusterinf data into groups based on similarity.
# Data partitioned into k clusters. 
# ML MODEL (clustering {numerical}, unsupervised) -> anmomaly detection, image compression.
from sklearn.cluster import KMeans

# ML model decision tree (classification, regression, supervised) -> spam detection, credit scoring, medical diagnosis.
from sklearn import tree

# Multi-layer Perceptron (MLP) is a type of neural network that consists of multiple layers of nodes.
# Each node applies a weighted sum of inputs followed by a non-linear activation function.
# Learns complex patterns in data by adjusting weights through backpropagation.
#     Backpropagation -> backwards through the network, starting from the output layer into the input layer.
# Models non-linear decision boundaries.
# ML MODEL (classification, regression, supervised) -> image recognition, speech recognition, time series prediction, etc.
from sklearn.neural_network import MLPClassifier

# Mean between precision and recall.
# Precision is the all of the items the model labels as positive.
# Recall is the all of the items that are actually positive.
# F1 = 2 * (precision * recall) / (precision + recall)
# Balance trade-off between false positives and false negatives.
from sklearn.metrics import f1_score
import warnings

# Scale input vectors individuall to have a unit norm (length of 1).
# Ensures that each sample has the same scale (magnitude).
# Useful for the direction of data points rather than their magnitude.
# Used by SVM, KMeans, and PCA.
from sklearn.preprocessing import normalize

# Collection library of statistical functions and probability distributions.
import scipy.stats
from scipy.stats import norm

# FEATURES

'''
    Feature descriptions:
        1. duration: duration of the connection in seconds.
        2. protocol_type: type of protocol.
        3. service: network type.
        4. flag: flag status.
        5. src_bytes: # of bytes transferred from source to destination.
        6. dst_bytes: # of bytes transferred from destination to source.
        7. land: 1 if connection is from the same host land=1, else 0.
        8. wrong_fragment: # of wrong fragments.
        9. urgent: # of urgent packets.
        10. hot: # of hot indicators.
        11. num_failed_logins: # of failed logins.
        12. logged_in: 1 if user is logged in, else 0.
        13. num_compromised: # of compromised accounts.
        14. root_shell: 1 if root shell is obtained, else 0.
        15. su_attempted: if "su root" accesses, su_attempted=1, else 0.
        16. num_root: # of accessed roots.
        17. num_file_creations: # of file creations.
        18. num_shells: # of shell prompt.
        19. num_file_creations: # of file creations.
        20. num_access_files: # of operations on access files.
        21. num_outbound_cmds: # of outbound commands.
        22. is_host_login: if host is is_host_login=1, else 0.
        23. is_guest_login: if login is guest is_guest_login=1, else 0.
        24. count: # of connections to the same host in last 2 seconds.
        25. srv_count: # of connections to the same service in last 2 seconds.
        26. serror_rate: % of connections with "SYN" errors.
        27. srv_serror_rate: % of connections with "SYN" errors.
        28. rerror_rate: % of connections with "REJ" errors.
        29. srv_serror_rate: % of connections with "REJ" errors.
        30. same_srv_rate: % of connections to the same service.
        31. diff_srv_rate: % of connections to different services.
        32. srv_diff_host_rate: % of connections to different hosts.
        33. dst_host_count: # of connections of same destination host and service.
        34. dst_host_srv_count: # of connections of same destination host and service.
        35. dst_host_same_srv_rate: % of connections having same destination host and service.
        36. dst_host_diff_srv_rate: % of connections having different service on current host.
        37. dst_host_same_src_port_rate: % of connections of current host having same src port.
        38. dst_host_srv_diff_host_rate: % of connections of same service and different hosts.
        39. dst_host_serror_rate: % of connections of current host having S0 error.
        40. dst_host_srv_serror_rate: % of connections of current host having of a service having S0 error.
        41. dst_host_rerror_rate: % of connections of current host host thatt rst error.
        42. dst_host_srv_rerror_rate: % of connections of connections of current host of service that have rst error.
        43. xAttack: Type of attack.
'''

feature_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","xAttack","level"]

import os
# Check current working directory
#print(os.getcwd())

# Load training and testing datasets
train = pd.read_csv("C:/Users/noahd/OneDrive/Desktop/IDS/Learning/KDDTrain+.txt")
test = pd.read_csv("C:/Users/noahd/OneDrive/Desktop/IDS/Learning/KDDTest+.txt")

# Check if the datasets are loaded correctly 
#print(train.head(20))
#print(train.tail(5))
#print(test.head(20))
#print(test.tail(5))

# Check training data dimensions
# We have 43 features and 125972 records
#print("Training data shape: ", train.shape)

# Statistical summary
train.columns = feature_names
test.columns = feature_names
#print(train.describe())
#print(train.info())

okDisplay = False

"""NUMERICAL FEATURES VISUALIZATION"""
# Identified numerical and assign them to numerical_features variable
numerical_features = [col for col in train.columns if train[col].dtype in ['int64', 'float64']]
#print("Numerical features: \n", numerical_features)

# After, we will display the frequency distribution of each numerical features using histograms.
if okDisplay:
    subsampled_data = train[numerical_features].sample(frac=0.1, random_state=42)
    # Calculate the number of rows and columns for subplots
    num_rows = (len(numerical_features) + 2) // 3 # Adjust the number of columns as needed
    num_cols = 3

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 6), facecolor='#F2F4F4')
    fig.subplots_adjust(wspace=0.2, hspace=0.5)

    # Flatten the axes array for easier indexing
    axes = axes.flatten()

    # Plot histograms for each numerical feature
    for i, feature in enumerate(numerical_features):
        h = sns.histplot(x=feature, kde=True, data=subsampled_data, bins=50, ax=axes[i])
        h.set_title(('Frequency Distribution of ' + feature).title(), fontsize=13)

    # Remove any empty subplots if the number of features is not a multiple of 3
    if len(numerical_features) % 3 != 0:
        for i in range(len(numerical_features), num_rows * num_cols):
            fig.delaxes(axes[i])

# Show the plot
#plt.tight_layout()
#plt.show()

"""CATEGORICAL FEATURES VISUALIZATION"""
# Identified categorical features and assign them to categorical_features variable
categorical_features = [col for col in train.columns if train[col].dtype == 'object' and col != 'xAttack']
#print("Categorical features: \n", categorical_features)

cat = train[categorical_features]
#print(cat.head())

# We can then use count plots to understand the distribution of categorical data.
# Set up the figure and axes.
if okDisplay:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='#F2F4F4')
    sns.countplot(x='protocol_type', data=train, order=train['protocol_type'].value_counts().index, ax=axes[0], palette='ch:.25')
    axes[0].set_title('Protocol Type Distribution', fontsize=12)
    axes[0].set_xlabel('Protocol Type')
    axes[0].set_ylabel('Count')

    # Distribution of 'flag'
    sns.countplot(x='flag', data=train, order=train['flag'].value_counts().index, ax=axes[1], palette='ch:.25')
    axes[1].set_title('Flag Distribution', fontsize=12)
    axes[1].set_xlabel('Flag')
    axes[1].set_ylabel('Count')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(16, 15))
    sns.set_style('darkgrid')

    # Distribution of 'service'
    ax = sns.countplot(y='service', data=train, order=train['service'].value_counts().index, palette='ch:.25')

    # Add percentage labels
    total = len(train['service'])
    for p in ax.patches:
        percentage = f'{100 * p.get_width() / total:.2f}%'
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height() / 2
        ax.annotate(percentage, (x, y), ha='left', va='center', fontsize=10, color='black')

    # Set title and adjust layout
    plt.title('Service Distribution', fontsize=15)
    plt.tight_layout()
    plt.show()

"""DATA PREPROCESSING"""
# In this step we have to analyze, filter, transform and encode data so that a
# ML algorithm can understand and work with the processed output.

"""Handling Missing Values"""
# Missing values are a recurrent problem in real-world datasets because real-life
# data has physical and manual limitations.

# Identify missing values.
missing_val_count_by_column = train.isnull().sum()
columns_with_missing_values = [col for col in train.columns if train[col].isnull().any()]
#print("Columns with missing values: ", columns_with_missing_values)

if okDisplay:
    plt.figure(figsize=(22,4))
    sns.heatmap((train.isna().sum()).to_frame(name='').T,cmap=sns.color_palette(["#283149", "#404B69", "#DBEDF3", "#DBDBDB", "#FFFFFF"]), annot=True,
                 fmt='0.0f').set_title('Count missing values', fontsize=18)
    plt.show()

"""Handling Duplicates"""
# Duplicate data refers to the presence of identical records in a dataset, which can
# distort the analysis and lead to incorrect conclusions. Removing duplicates is crucial
# for accurate analysis and modeling.

#print(train.duplicated().sum()) # Check for duplicates in the training set
#print(test.duplicated().sum()) # Check for duplicates in the testing sets

"""Exploring target (xAttack)"""
"""
    Denial of Service attacks:
        - apache2
        - back
        - land
        - neptune
        - mailbomb
        - pod
        - processtable
        - smurf
        - teardrop
        - udpstorm
        - worm
    Probe attacks:
        - ipsweep
        - mscan
        - nmap
        - portsweep
        - saint
        - satan
    Privilege escalation attacks:
        - buffer_overflow
        - loadmodule
        - perl
        - ps
        - rootkit
        - sqlattack
        - xterm
    Remote access attacks:
        - ftp_write
        - guess_passwd
        - httptunnel
        - imap
        - multihop
        - named
        - phf
        - sendmail
        - snmpgetattack
        - snmpguess
        - spy
        - warezclient
        - warezmaster
        - xclock
        - xsnoop
"""

attack_mapping = {
    'neptune': 'DoS', 'back':'DoS', 'land': 'DoS', 'pod': 'DoS', 'smurf': 'DoS', 'teardrop': 'DoS', 'mailbomb': 'DoS', 'apache2': 'DoS',
    'processtable': 'DoS', 'udpstorm': 'DoS', 'worm': 'DoS',
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R', 'ps': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R',
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L', 'phf': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L',
    'warezmaster': 'R2L', 'sendmail': 'R2L', 'named': 'R2L', 'snmpgetattack': 'R2L', 'snmpguess': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L',
    'httptunnel': 'R2L',
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe', 'mscan': 'Probe', 'saint': 'Probe'
}
train['xAttack'] = train['xAttack'].replace(attack_mapping)
test['xAttack'] = test['xAttack'].replace(attack_mapping)
#print(train.head(5))

# Checking the distribution of attack families.
if okDisplay:
    # Set the style and color palette for the plot
    sns.set(style="darkgrid")
    colors = sns.color_palette('pastel')

    # Plot using Seaborn
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x='xAttack', data=train, palette='ch:.25')

    # Show percentages on top of the bars
    total = len(train['xAttack'])
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height() / total)
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(percentage, (x, y), ha='center', va='bottom', fontsize=10, color='black')

    plt.title('Distribution of the Normal, DoS, U2R, R2L and Probe attacks in the Dataset')
    plt.show()

"""Check the cardinality of the categorical features"""
# Cardinality refers to the number of different values in a variable. As we will create
# dummy variables from the categorical variables later on, we need to check whether
# there are variables with many distinct values. We should handle these variables
# differently as they would result in many dummy variables.

# Get number of unique entries in each column with categorical data
object_nunique = list(map(lambda col: train[col].nunique(), categorical_features))
d = dict(zip(categorical_features, object_nunique))
# Print number of unique entries by column, in ascending order
print(sorted(d.items(), key=lambda x: x[1]))

# Plot the unique values
if okDisplay:
    unique = train[categorical_features].nunique()
    plt.figure(figsize=(20, 10))
    unique.plot(kind='bar', color=['#404B69', '#5CDB95', '#ED4C67', '#F7DC6F'], hatch='//')
    plt.title('Unique elements in each categorical features')
    plt.ylabel('Count')
    for i, v in enumerate(unique.values):
        plt.text(i, v+1, str(v), color='black', ha='center')
    plt.show()

""" Univariate Analysis """
# In every technique which comes under the hood of Univariate Selection, every feature
# is individually studied and the relationship it shares with the target variable is
# taken into account.

# We will see at first the distribution of the numerical values data using mean, median, Q1 and Q3.

# Check variables distribution
data_num = train[numerical_features].sample(frac=0.1, random_state=42)
num_rows = (len(numerical_features) + 2) // 3  # Adjust the number of columns as needed

if okDisplay:
    fig, axes = plt.subplots(nrows=num_rows, ncols=3, figsize=(26, num_rows * 6))
    colors = ["#283149"]

    # Loop through each column and plot distribution
    for i, column in enumerate(numerical_features):
        sns.histplot(x=column, data=data_num, color=colors[i%1], ax=axes[i//2, i%2], kde=True)

        # Add vertical lines for mean, median, Q1 and Q3
        axes[i//2, i%2].axvline(x=train[column].median(), color='#e33434', linestyle='--', linewidth=2, label='Median')
        axes[i//2, i%2].axvline(x=train[column].quantile(0.25), color='orange', linestyle='--', linewidth=2, label='Q1')
        axes[i//2, i%2].axvline(x=train[column].quantile(0.75), color='#177ab0', linestyle='--', linewidth=2, label='Q3')

        # Add text box with important statistics
        median = train[column].median()
        q1 = train[column].quantile(0.25)
        q3 = train[column].quantile(0.75)
        iqr = q3 - q1
        axes[i//2, i%2].text(0.95, 0.95, 'Mean: {:.2f}\nMedian: {:.2f}\nQ1: {:.2f}\nQ3: {:.2f}\nIQR: {:.2f}\nMax: {:.2f}'.format(
                            train[column].mean(), median, q1, q3, iqr, train[column].max()),
                            transform=axes[i//2, i%2].transAxes, fontsize=10, va='top', ha='right')
        
        # Add legend
        axes[i//2, i%2].legend(loc = "upper left")

        # Set title of subplot
        axes[i//2, i%2].set_title('Distribution of ' + column)

    # Remove any empty subplots if the number of features is not a multiple of 3
    if len(numerical_features) % 3 != 0:
        for i in range(len(numerical_features), num_rows * num_cols):
            fig.delaxes(axes[i // 3, i % 3])

    fig.suptitle('Distribution of Numerical Variables', fontsize=16)
    fig.tight_layout()

#print('Done')

# Variance
"""
    Variance is the measure of change in a given feature. For example, if all the samples in a feature have
    the same value, it would mean that the variance of that feature is zero. It's essential to understand
    that a column which doesn't have enough variance is as good as a column with all 'nan' or missing values.
    If there's no change in the feature, it's impossible to derive any pattern from it. So, we check the variance
    and eliminate any feature that shows low or no variation.

    Variance thresholds might be a good way to eliminate features in datasets, but in cases where there are
    minority calsses (say, 5% -s amd 95% 1s), even good features can have very low variance and still end up
    being very strong predictors. So, be advised -- keep the target ratio in mind and use correlation methods
    before eliminating features solely based on variance.
"""

""" VARIANCE THERESHOLD """
# Calculate the variance of each feature
# Features with very low variance may not provide much information and can be dropped.
quantitative_data = train.select_dtypes(include='number')
print("Quantitative data shape: ", quantitative_data.shape)

from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=0.01)  # Set a threshold for variance
sel.fit(quantitative_data)
mask=sel.get_support()  # Get the mask of selected features
reduced_df = quantitative_data.loc[:, mask]  # Apply the mask to the DataFrame
#print(mask)
#print(reduced_df.shape)

feature_variance = sel.variances_
total_variance = sum(feature_variance)
feature_variance_percentage = (feature_variance / total_variance) * 100

if okDisplay:
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=quantitative_data.columns, y=feature_variance_percentage, marker='o', color='blue', label='Variance Percentage', linestyle='-')
    plt.axhline(y=0.01 / total_variance * 100, color= 'red', linestyle='--', label=f'Threshold ({0.01 * 100:.2f}%)')
    plt.xlabel('Features')
    plt.ylabel('Variance (%)')
    plt.title('Feature Variance and Threshold (as Precentage of Total Variance)')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True)
    plt.show()

columns_to_drop_variance = quantitative_data.columns[~sel.get_support()]
#print('Columns that can be dropped based on variance threshold:', columns_to_drop_variance.tolist())

columns_to_keep = train.columns.tolist()
test = test[columns_to_keep]
#print('Columns to keep:', columns_to_keep)
#print('Test shape:', test.shape)

# Correlation
"""
    Correlation is a univariate analysis technique. It detects linear relationships between two variables. Think
    of correlation as a measure of proportionality, which simply measures how the increase or decrease of a variable
    affects the other variable.
"""
""" CORRELATION  """
# FInding Correlation among Features
corr = train.select_dtypes(include='number').corr()
if okDisplay:
    f, ax = plt.subplots(figsize=(12, 10))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={'shrink': .75})
    plt.title('Correlation Matrix Heatmap')
    plt.show()

# Calculate the correlation matrix for our features.
# Identify pairs of highly correlated features
# If two features are highly correlated, you might consider keeping only one of them

plt.figure(figsize=(40,35))
corr_df = train.select_dtypes(include='number').corr().abs()

# .triu() keeps the upper triangle of a mtrix (including the diagonal)
mask = np.triu(np.ones_like(corr_df, dtype=bool)) # Creates a matrix of True values

# Applies the mask: wherever the mask is True, the value is replaced with NaN.
# Result: the upper triangle of corr_df is hidden, leaving just the lower triangle (excluding duplicates)
tri_df = corr_df.mask(mask)

# The upper and lower triangles are mirror images (because correlation is symmetric)
# By hiding the upper triangle, you avoid visual clutter or duplication when plotting.
to_drop = [c for c in tri_df.columns if any(tri_df[c] >= 0.9)]
print('Columns that can be dropped using correlation:', to_drop)

# Now, drop high correlation features
train = train.drop(columns=to_drop, axis=1)
test = test.drop(columns=to_drop, axis=1)
print(train.shape)
print(test.shape)

""" BINARY CLASSIFICATION """
# Using an encoder (BINARY LABELS THROUGH LabelEncoder)
attack_n = []
for i in train.xAttack:
    if i == 'normal':
        attack_n.append('normal')
    else:
        attack_n.append('attack')

train['xAttack'] = attack_n
#print(train['xAttack'])

attack_n = []
for i in test.xAttack:
    if i == 'normal':
        attack_n.append('normal')
    else:
        attack_n.append('attack')

test['xAttack'] = attack_n
#print(test['xAttack'])

train.loc[train['xAttack'] == 'normal', 'xAttack'] = 0
train.loc[train['xAttack'] != 0, 'xAttack'] = 1
#print(train['xAttack'])

x_train = train.drop(['xAttack', 'level'], axis=1)
y_train = train['xAttack']
y_train = y_train.astype('int')
x_train = train.select_dtypes(include='number')

test.loc[test['xAttack'] == 'normal', 'xAttack'] = 0
test.loc[test['xAttack'] != 0, 'xAttack'] = 1
#print(test['xAttack'])

# Now we can visualize a pie chart to see the distribution of normal and abnormal labels of our binary data
label_counts = train.xAttack.value_counts()

if okDisplay:
    colors = ['#66b3ff', '#99ff99', '#ffcc99', 'c2c2f0', 'ffb3e6']

    plt.figure(figsize=(10, 10))
    plt.pie(label_counts, labels=label_counts.index, autopct='%0.2f%%', colors=colors, shadow=True, startangle=140)
    plt.title('Pie Chart Distribution of Labels')
    plt.legend(title='Labels', loc='upper right')
    plt.show()

""" CHI-SQUARE """

"""
    Chi-square is a statistical tool, or test, which can be used on groups of categorical features to evaluate the
    likelihood of association, or correlation, with the help of frequency distributions.
"""

selector = SelectKBest(chi2, k=10)
selector.fit(x_train, y_train)

selected_features = train.columns[selector.get_support(indices=True)]
feature_scores = selector.scores_[selector.get_support(indices=True)]

if okDisplay:
    plt.figure(figsize=(12, 6))
    bar_colors = sns.color_palette('ch:.25', n_colors=len(selected_features))
    sns.barplot(x=feature_scores, y=selected_features, palette=bar_colors)

    for idx, score in enumerate(feature_scores):
        plt.text(score + 0.02, idx, f'{score:.2f}', ha='left', va='center')

    plt.xlabel('Chi-squared Score')
    plt.ylabel('Features')
    plt.title('Top 10 Features Selected by SelectKBest with Chi-squared Test')
    plt.show()

# Logistic Regression
"""
    Logistic regression is a statistical ML method used for binary classification
        - predicting one of two possible outcomes
    
    Unlinke linear regression, logistic regression predicts a probability between 0 and 1
    using the logistic (sigmoid) function

        Common use cases:
            - Spam detection
            - Medical diagnosis
            - Fraud detection
            - Customer churn prediction
"""

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)

"""
    lr.coef_ (the array of coefficient learned by your logistic regression model(lr). )
    zip() -> pairs each feature name with its corresponding absolute coefficient

    example:
        'feature1' : coefficient

    useful for:
        1. Interpreting feature importance - bigger absoulte value means stronger influence on the prediction
        2. Feature Selection - identifying the most relevant inputs.
"""
coef_dict = dict(zip(x_train.columns, abs(lr.coef_[0])))
#print(coef_dict)

"""
    RFE stands for Recursive Feature Elimination
        - feature selection method

        How it works:
            1. Train the model on all features
            2. Rank features by important (using coefficients)
            3. Remove the least important feature.
            4. Repeat until specified number of features is left.
"""
from sklearn.feature_selection import RFE
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=2, verbose=1)
rfe.fit(x_train, y_train)

"""
    Use .ranking_ when:
        1. You want to see the relative importance of all features
        2. You might want to experiment with different features thresholds.
        3. You're debugging or doing analysis

    Use .support_ when:
        1. You want a quick True/False mask showing which features were selected.
        2. You want to immediately filter your dataset to just the selected features.

    In practice:
        .support_ is most useful after you've already told RFE how many features to select (n_features_to_select=X)
        .ranking_ is most useful if you want to understand the full picture or decide how many top features to keep
"""
rfe_dict = dict(zip(x_train.columns, rfe.ranking_))
rfe_dict2 = dict(zip(x_train.columns, rfe.support_))
#print(rfe_dict)
#print(rfe_dict2)
# for elements in rfe_dict2:
#     print(f'{elements}: {rfe_dict2[elements]}')

# Now we drop the flag, service, protocol feature
train = train.drop(columns=['flag', 'service', 'protocol_type'], axis=1)
test = test.drop(columns=['flag', 'service', 'protocol_type'], axis=1)

""" OUTLIERS """
"""
    Investigating Outliers is an essential step in data analysis because they can significantly affect the
    statistical memasures used to describe a dataset. Outliers are observations that differ significantly from
    other observations in the same dataset and can result from measurement errors, sampling issues, or genuine
    differences in the population. Identifying and dealing with outliers can help to improve the accuracy and
    reliability of statistical models and results, leading to more informed decisions and better outcomes.
    Therefore, outlier investigation is a cruical step in any data analysis process.
"""

# Before we define outliers, we will need to drop protocol_type feature since it's a categorical feature
# Then, we us a boxplot to identify outliers within each column.
if True:
    plt.figure(figsize=(15, 10))
    sns.set_style('darkgrid')
    sns.boxplot(data=quantitative_data)
    plt.xticks(rotation=98)
    #plt.show()

# We also defined boxplots to define outliers seperately within each column
def plot_numerical_features_boxplots(data, columns_list, title):
    num_features = len(columns_list)
    cols = 3
    rows = (num_features + cols - 1) // cols

    sns.set_style('darkgrid')
    fig, axs = plt.subplots(rows, cols, figsize=(18, 7 * rows), sharey=True)
    fig.suptitle(title, fontsize=25, y=1)

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.5)

    axs = axs.flatten()

    # Initialize an empty DF to collect outliers
    outliers_df = pd.DataFrame(columns=['Column', 'Outlier_index', 'Outlier_values'])

    for i, col in enumerate(columns_list):
        sns.boxplot(x=data[col], color='#404869', ax=axs[i])
        axs[i].set_title(f'{col} (skewness: {round(float(data[col].skew()), 2)})', fontsize=12)

        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1

        # In Python, | symbol means logical OR when used with arrays or Series, like in pands or numpy.
        outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR)))
        outliers_index = data[outliers].index.tolist()
        outliers_values = data[col][outliers].tolist()

        # Append outliers to the DF
        outliers_df = pd.concat([outliers_df, pd.DataFrame({'Column': [col], 'Outlier_index': [outliers_index],
                                                            'Outlier_values': [outliers_values]})], ignore_index=True)
        
        axs[i].plot([], [], 'ro', alpha=0.5, label=f'Outliers: {outliers.sum()}')
        axs[i].legend(loc='upper right', fontsize=10)

    # Hide empty subplots (if any)
    for i in range(num_features, rows * cols):
        axs[i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust the layout
    return outliers_df

# Example usage
outliers_df_result = plot_numerical_features_boxplots(data=quantitative_data, columns_list=quantitative_data.columns,
                                                      title='Boxplots for Outliers')

"""
    We defined blow functions to remove outliers from both training and test sets. But they are momental until
    we define better functions to deal with outliers.
"""

# Function ti identify and remove outleirs based on IQR
# k is a multiplier that controls how far from the IQR you're willing to go before calling a value an outlier
# Anything outside of the IQR is considered an outlier
def remove_outliers(data, k=1.5):
    outliers = pd.DataFrame()

    for column in data.columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - k * IQR
        upper_bound = Q3 + k * IQR

        column_outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        outliers = pd.concat([outliers, column_outliers])

    return outliers

# Identify and remove outliers
outliers = remove_outliers(quantitative_data)

# The ~ symbol in this line meaans logical NOT (bitwise negation operator in Python)
# Used with a Boolean Series or array (like in pandas).
quantitative_data = quantitative_data[~quantitative_data.index.isin(outliers.index)]

# Diplay the first few rows of the dataset without outliers

# The .loc[] in pands is used for label-based indexing, it lets you access rows and columns by their labels,
# not by position.
# Usage example: df.loc[row_label, column_label]
# But saying df.loc[:, ...] tells pandas to give me all rows (:) and only the columns I specify after the comma
filtered_data_no_outliers = quantitative_data.loc[:, (quantitative_data != 0).any(axis=0)]
print(filtered_data_no_outliers.head())
#filtered_data_no_outliers.columns()

# Duration feauture was an important feature in chi-squared and Logistic regression
train.head(10)
test.head(10)

"""
    Generating Datasets
"""

# The original NSL dataset without any transformation of the numerical values
print('\nThe original NSL dataset without any transformation of the numerical values')
print(train.info())

d_raw_train = train.copy()
d_raw_test = test.copy()
print(f'Raw dataset train shape: {d_raw_train.shape}')

# xAttack is an object d-type, we will transform it into numerical values

train_target = d_raw_train['xAttack']
train_target = train_target.astype('int')
d_raw_train = d_raw_train.drop('xAttack', axis=1)

test_target = d_raw_test['xAttack']
test_target = test_target.astype('int')

print('Train target:\n', train_target)
print('\nTest target:\n', test_target)

d_raw_train['xAttack'] = train_target
d_raw_train_normal = d_raw_train[d_raw_train['xAttack'] == 0]

# Drop the column from the DF and it in place without needing to reassign it.
# axis=1 tells pandas to drop columns (not rows)
#   axis=0 -> drop rows
#   axis=1 -> drop columns
#   inplace=True -> make the change directly in the existing DF
d_raw_train_normal.drop(['xAttack'], inplace=True, axis=1)
d_raw_train.drop(['xAttack'], inplace=True, axis=1)
print(d_raw_train_normal.head())

# d_raw_probs
"""
    We apply the FGMPM to the original NSL dataset values and change each feature value for the
    occurance probability of each feature in the normal model.

    What is FGMPM? (need to ask professor)
        -> GMM in a custom, non-standard way. (Gaussian Mixture Model) maybe?

    GMM is a probabilistic model that assumes data is generated from a mixture of several Gaussian
    distributions (normal distributions), each representing a cluster or component in the data.

        Mainly used for:
            1. Clustering (like k-means, but more flexible)
            2. Anomaly detection
            3. Density estimation

        How it works:
            1. Assumes your data is generated by several Gaussian distributions
            2. Each Gaussian has:
                a. A mean (center)
                b. A covariance (spread/shape)
                c. A weight (how much of the data it explains)
            3. GMM uses the Expectation-Maximization (EM) algorithm to learn:
                a. How many Guassian best explains your data
                b. The paraneters of each Gaussian
    
    For the purposes of this IDS, a custom-made GMM is made
"""

# Threshold paramater is not being used
def GMM_Row_Transform(data, values, threshold):
    # List to store rarity probabilities for each feature
    probs = []

    for idx in range(len(data.columns)):
        mean = np.array(data.iloc[:, idx]).mean() # Mean
        std = np.array(data.iloc[:, idx]).std() # Standard deviation

        # Z-score measures how many stds a value is from the mean
        z_score = (values[idx] - mean)/std

        # P-value gives the probability of observing a value as extreme as the one you're testing, assumming
        # a certain null hypothesis (Gaussian distribution data), use z-scores to find p-values
        prob = (1-norm.cdf(z_score)) * 100
        probs.append(prob)
    
    return probs

def GMM_Matrix_Transform(origin_data, data, threshold):
    matrix = []

    # tdqm is a Python progress bar library that makes loops easier to monitor
    for i in tqdm(range(len(data))):
        row = GMM_Row_Transform(origin_data, data.iloc[i, :], threshold)

        matrix.append(row)

    return matrix

def GMM_Transform(data):
    GMM_Transformed = []

    for i in tqdm(range(len(data))):
        row = []
        for idx in range(len(data[i])):
            mean = np.array(data[:, idx]).mean()
            std = np.array(data[:, idx]).std()
            z_score = (data[i, idx] - mean) / std
            prob = (1 - norm.cdf(z_score)) * 100
            row.append(prob)
        
        GMM_Transformed.append(row)

    return np.array(GMM_Transformed)

# Used for the MODELING section (Voting)
def GMM_vote(data, values, threshold):
    no = 0
    for idx in range(len(data.columns)):
        mean = np.array(data.iloc[:, idx]).mean()
        std = np.array(data.iloc[:, idx]).std()
        z_score = (values[idx] - mean) / std
        prob = (1 - norm.cdf(z_score)) * 100
        if prob <= threshold:
            no += 1

    return no

# Threshold is not being used by GMM_Row_Transform, which is being called by GMM_Matrix_Transform
# So, I don't know why there is a parameter named Threshold.
d_raw_probs_train = pd.DataFrame(GMM_Matrix_Transform(d_raw_train, d_raw_train, 50))
d_raw_probs_test = pd.DataFrame(GMM_Matrix_Transform(d_raw_train, d_raw_test, 50))

# Only normal attacks
d_raw_probs_train_normal = pd.DataFrame(GMM_Matrix_Transform(d_raw_train_normal, d_raw_train_normal, 50))

""" d_raw_pca """
"""
    Principal Component Analysis (PCA) is a technique used to reduce the dimensions of a dataset while
    minimizing information loss. It does this by combining information from all variables into
    Principal Components (PCs) that are uncorrelated with each other.

    PCA allows us to reduce the number of dimensions while retaining as much information as possible.
    We can choose to discard some of the PCs and use the remaining ones as our variables. This results
    in a dataset with fewer dimensions, but without any significant loss of information. Additionally,
    the new variables created through PCA are uncorrelated, which can be useful for downstream analysis.

        - What does reducing dimensionality mean?
            Reducing dimensionality of a dataset, you take your data that has many features (variables,
            columns) and transform it into a new space with fewere features -- while trying to keep as
            much important information as possible.

            -> Simplifies the data: fewer features means easier to analyze and visualize.
            -> Remove noise/redundancy: some features might be correlated or not add useful info.
            -> Speed up algorithm: fewer features means faster computations
            -> Prevent overfitting: less complexity can help models generalize better

    Reducing the number of variables of a dataset naturally comes at the expense of accuracy, but the
    trick in dimensionality reduction is to trade a little accuracy for simplicity because smaller data
    sets are easier to explore and visualize and make analyzing data much easier and faster for machine
    learning algorithms without extraneous variables to process.

    We use PCA since PCA does not need labels compared to LDA.
        - Unsupervised learning or exploratory analysis: PCA is more suitable for unsupervised scenarios
        - It focuses on capturing overall variance
        - Useful on data with more classes than features
"""
def PCA_transformation(data, n_components=20):
    pca = PCA(n_components=n_components) # Create PCA object
    x_train_reduced = pca.fit_transform(data)

    # An array where each value represents how much variance (information) each PC explains relative
    # to the total variance in the original data.
    explained_variance_ratio = pca.explained_variance_ratio_

    # This computes the cumulative sum of explained variance ratios
    # It tells you how much total variance is explained by the first K components cumulatively
    cumulative_variance = explained_variance_ratio.cumsum()

    return x_train_reduced, cumulative_variance

d_raw_pca_train, x1 = PCA_transformation(d_raw_train)
d_raw_pca_test, x2 = PCA_transformation(d_raw_test)
print(d_raw_pca_test.shape)

if okDisplay:
    plt.plot(x1)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.show()

    plt.plot(x2)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.show()

d_raw_pca_train_normal, _ = PCA_transformation(d_raw_train_normal)

""" d_norm """
"""
    The original NSL dataset with the normal training values normalized to the range [0-1] and the
    remaining values normalized according to the previous scaler.

    Data scaling is necessary when the range of values differs across columns. By scaling the data, we
    ensure that each column has the same range or standardization of values. Standardization is important
    because higher scales may result in greater variance or covariance values, which can lead to bias.
    Therefore, we will begin by standardizing the features.

        -> Brings features to a common scale so that no feature dominates because of its magnitude
        -> Improve algorithm performance since a lot of ML models assume that features are on
           similar scales and work better or converge faster when data is scaled.
        -> Avoid bias in distance-based methods since it can be heavily influenced by features with
           large ranges if you do not scale
        -> Stabilize numerical computations
"""

def normalizing(data):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    data = pd.DataFrame(data_scaled, columns=data.columns)

    return data

d_norm_train = normalizing(d_raw_train)
d_norm_test = normalizing(d_raw_test)
d_norm_train_normal = normalizing(d_raw_train_normal)

d_norm_pca_train,_ = PCA_transformation(d_norm_train)
d_norm_pca_test,_ = PCA_transformation(d_norm_test)
d_norm_pca_train_normal, dim = PCA_transformation(normalizing(d_raw_train_normal))

""" d_raw_pca_probs """
"""
    We apply the FGMPM to the uncorrelated version of the original dataset and obtain the occurrence
    probabilities for this uncorrelated values of the features.
"""
def PCA_transformation_2(data):
    pca = PCA(20)
    data = pca.fit_transform(data)
    return pd.DataFrame(data)

d_raw_pca_probs_train= PCA_transformation_2(d_raw_train)
d_raw_pca_probs_train = pd.DataFrame(GMM_Matrix_Transform(d_raw_pca_probs_train, d_raw_pca_probs_train, 50))

d_raw_pca_probs_test = PCA_transformation_2(d_raw_test)
d_raw_pca_probs_test = pd.DataFrame(GMM_Matrix_Transform(d_raw_pca_probs_train, d_raw_pca_probs_test, 50))

d_raw_pca_probs_train_normal = PCA_transformation_2(d_raw_train_normal)
d_raw_pca_probs_train_normal = pd.DataFrame(GMM_Matrix_Transform(d_raw_pca_probs_train_normal, d_raw_pca_probs_train_normal, 50))

""" d_norm_probs """
"""
    We apply the FGPM to the normalized version of the dataset.
"""

d_norm_probs_train = normalizing(d_raw_train)
d_norm_probs_train = pd.DataFrame(GMM_Matrix_Transform(d_norm_probs_train, d_norm_probs_train, 50))

d_norm_probs_test = normalizing(d_raw_test)
d_norm_probs_test = pd.DataFrame(GMM_Matrix_Transform(d_norm_probs_train, d_norm_probs_test, 50))

d_norm_probs_train_normal = normalizing(d_raw_train_normal)
d_norm_probs_train_normal = pd.DataFrame(GMM_Matrix_Transform(d_norm_probs_train_normal, d_norm_probs_train_normal, 50))

""" d_norm_pca_probs """
"""
    The occurrence probabilities of the uncorrelated features of the normalized dataset.
"""
d_norm_pca_probs_train = normalizing(d_raw_train)
d_norm_pca_probs_train = PCA_transformation_2(d_norm_pca_probs_train)
d_norm_pca_probs_train = pd.DataFrame(GMM_Matrix_Transform(d_norm_pca_probs_train, d_norm_pca_probs_train, 50))

d_norm_pca_probs_test = normalizing(d_raw_test)
d_norm_pca_probs_test= PCA_transformation_2(d_norm_pca_probs_test)
d_norm_pca_probs_test = pd.DataFrame(GMM_Matrix_Transform(d_norm_pca_probs_train, d_norm_pca_probs_test, 50))


d_norm_pca_probs_train_normal = normalizing(d_raw_train_normal)
d_norm_pca_probs_train_normal = PCA_transformation_2(d_norm_pca_probs_train_normal)
d_norm_pca_probs_train_normal = pd.DataFrame(GMM_Matrix_Transform(d_norm_pca_probs_train_normal, d_norm_pca_probs_train_normal, 50))

""" MODELING """
"""
    Build modles and apply them later on each of generated datasets

    Proposed voting scheme method for anomaly detection that can only be applied to the
    probability datasets.
"""

# def GMM_vote(data, values, threshold):
#     no = 0
#     for idx in range(len(data.columns)):
#         mean = np.array(data.iloc[:, idx]).mean()
#         std = np.array(data.iloc[:, idx]).std()
#         z_score = (values[idx] - mean) / std
#         prob = (1 - norm.cdf(z_score)) * 100
#         if prob <= threshold:
#             no += 1
#     return no

"""
Function is perfomring unsupervised anomaly detection using the GMM-based voting
system. It analyzes each row in the test data and votes on whether its an anomaly
based on how many of its features appear abnormal compared to the training data.
   data_train   -> normal data
   data_test    -> new data checking for anomalies
   min_abnormal_features -> how many individual features need to be considered "abnormal"
                            before the whole row is flagged as an anomaly.
   threshold    -> used by GMM_vote to decide whether a feature value is abnormal
"""
def voting(data_train, data_test, min_abnormal_features=10, threshold=50):
    preds = []
    for idx in tqdm(range(len(data_test))):
        values = list(data_test.iloc[idx,:])    # Gets all the feature values from the current test sample (row) as a list
        no = GMM_vote(data_train, values, threshold) # Gets the amount of anomaly features exist for the row.
        if no > min_abnormal_features:
            preds.append(1)
        else:
            preds.append(0)

    return preds


voting_d_raw_probs_preds = voting(d_raw_probs_train_normal, d_raw_probs_test, min_abnormal_features = 10, threshold = 50)
voting_d_raw_pca_probs_preds = voting(d_raw_pca_probs_train_normal, d_raw_pca_probs_test, min_abnormal_features = 10, threshold = 50)
voting_d_norm_probs_preds = voting(d_norm_probs_train_normal, d_norm_probs_test, min_abnormal_features = 10, threshold = 50)
voting_d_norm_pca_probs_preds = voting(d_norm_pca_probs_train_normal, d_norm_pca_probs_test, min_abnormal_features = 10, threshold = 50)

""" DECISION TREE """
"""
    Why use a decision tree for anomaly detection?
    1. Captures Nonlinear Patterns:
        - Decision trees naturally model nonlinear relationships and interactions between features
          which is useful if anomalies are defined by complex combinations of values.
    2. Interpretable Rules:
        - Tree generate clear decision paths. Very helpful when explaining why something is
          considered abnormal.
    3. Handles Mixed Data:
        - They handle both categorical and numerical features natively without needing special
          preprocessing like scaling.
    4. Isolation Forest (Tree-based Anomaly Detector):
        - A specialized ensemble of decision trees designed specifically for anomaly detection.
        - It works by randomly splitting features anomalies are easier to isolate and usually require
          fewer splits, so they appear in shallower branches.
        - Fast, scalable, usable on high-dimensional datasets.
    5. Flexible Thresholding

    NEEDS LABELED DATA. MAY OVERFIT ON SMALL OR NOISY DATASETS 
"""
# default decision tree classifier
def dt_model(train_data, test_data):
    model = tree.DecisionTreeClassifier()
    model.fit(train_data, train_target)
    preds = model.predict(test_data)
    return preds

dt_d_raw_preds = dt_model(d_raw_train, d_raw_test)
dt_d_raw_pca_preds = dt_model(d_raw_pca_train, d_raw_pca_test)
dt_d_raw_pca_probs_preds = dt_model(d_raw_pca_probs_train, d_raw_pca_probs_test)
dt_d_norm_preds = dt_model(d_norm_train, d_norm_test)
dt_d_norm_probs_preds = dt_model(d_norm_probs_train, d_norm_probs_test)
dt_d_norm_pca_preds = dt_model(d_norm_pca_train, d_norm_pca_test)
dt_d_norm_pca_probs_preds = dt_model(d_norm_pca_probs_train, d_norm_pca_probs_test)

""" Support Vector Machine (SVM) """
"""
    We will use this algorithm with the objective of obtaining a membership decision
    boundary for only one class of data.

    Why use SVM in anomaly detection?
    1. Boundary-based detection (One-class SVM)
        - tries to find the smallest boundary that encloses the majority of the data
          points, assuming they are normal.
    2. Works with Unlabeled Data
        - Can be trained with only "normal" data, making it useful in real-world cases
          where anomaly labels are rare or unavailable.
    3. Effective in High-Dimensional Spaces
        - Can use kernels (RBF) to model nonlinear decision boundaries, which is useful
          when anomalies are not linearly separable from normal data.
    4. Customizable Sensitivity
        - The nu parameter controls how strict the model is -- i.e., the proportion of data
          it allows to be outside the boundary.
    5. Built on solid mathematical optimization, One-cclass SVM maximizes margin around the
       normal data while penalizing outliers.
"""

def SVM_model(train_data, test_data):
    model = SVC()
    model.fit(train_data, train_target)
    preds = model.predict(test_data)
    return preds

svm_d_raw_preds = SVM_model(d_raw_train, d_raw_test)
svm_d_raw_probs_preds = SVM_model(d_raw_probs_train, d_raw_probs_test)
svm_d_raw_pca_preds = SVM_model(d_raw_pca_train, d_raw_pca_test)
svm_d_raw_pca_probs_preds = SVM_model(d_raw_pca_probs_train, d_raw_pca_probs_test)
svm_d_norm_preds = SVM_model(d_norm_train, d_norm_test)
svm_d_norm_probs_preds = SVM_model(d_norm_probs_train, d_norm_probs_test)
svm_d_norm_pca_preds = SVM_model(d_norm_pca_train, d_norm_pca_test)
svm_d_norm_pca_probs_preds = SVM_model(d_norm_pca_probs_train, d_norm_pca_probs_test)

""" Multi Layer Perceptron (MLP) """
"""
    MLP -> type of feedforward neural network
        -> great for anomaly detection, especially in more complex or high-dimensional.
    
    1. Unsupervised or Supervised Use
        a. Unsupervised
            = An MLP-based autoencoder learns to recontruct normal data
            = Anomalies will have high reconstruction error because they differ from normal
              patterns.
            (loss = | original - reconstructed |)
        
        b. Supervised
            = If you have labeled data (normal vs anomaly), you can train an MLP classifier
              to directly predict anomaly labels.

    2. Can model complex patterns
        - MLPs can learn nonlinear relationships in data that simpler models may miss.
        - This is especially helpful when anomalies are subtle on dependent on interactions
          between features.

    3. Scales to High-Dimensional Data
        - MLPs handle large numbers of feature better than traditional models.
        - Works well in domains like cybersecurity, manufacturing, finance, bioinformatics.
    
"""
"""
    A simple multilayer perceptron with a hidden layer of 100 neurons and an output layer
    with 2 cells: attack or non-attack.
"""

def MLP_Model(train_data, test_data):
    model = MLPClassifier(max_iter=500).fit(train_data, train_target)
    preds = model.predict(test_data)
    return preds

mlp_d_raw_preds = MLP_Model(d_raw_train, d_raw_test)
mlp_d_raw_probs_preds = MLP_Model(d_raw_probs_train, d_raw_probs_test)
mlp_d_raw_pca_preds = MLP_Model(d_raw_pca_train, d_raw_pca_test)
mlp_d_raw_pca_probs_preds = MLP_Model(d_raw_pca_probs_train, d_raw_pca_probs_test)
mlp_d_norm_preds = MLP_Model(d_norm_train, d_norm_test)
mlp_d_norm_probs_preds = MLP_Model(d_norm_probs_train, d_norm_probs_test)
mlp_d_norm_pca_preds = MLP_Model(d_norm_pca_train, d_norm_pca_test)
mlp_d_norm_pca_probs_preds = MLP_Model(d_norm_pca_probs_train, d_norm_pca_probs_test)

""" K-Means (Euclidean Distance) """
"""
    The well known K-Means algorithm using the anomaly detection approach with the squared euclidean distances.
"""
"""
    K-Means clustering uses Euclidean distance to measure similarity between data points and centroids.
    Given two points, the Euclidean distance is used to:
        1. Assign points to clusters - each data point is assigned to the nearest centroid (based on Euclidean distance)
        2. Repeat until covergence:
            - Assing each data point to the nearest centroid using Euclidean distance.
            - Recalculate centroids as the mean of the assigned points.

        ** Feature scaling is crucial. Use normalization or standardization if your features are on different scales,
           otherwise distance calculations will be biased.

    USED FOR UNSUPERVISED
        WHY?
            -> No label are provided during training
            -> The algorithm tries to discover patterns or groupings in the data based solely on feature similarity.
            -> It learns from the data.

    CAN BE OVERFITTED
        Overfitting happens when a model captures not only the underlying structure in the data but also the noise or
        random fluctations, reducing its ability to generalize new or unseen data.

            -> Performing well on training data but poorly on test data

        FOR K-MEANS:
            - Too many clusters (k is too high)
                    = If you set k equal to the number of data points, K-Means perfectly "clusters" every point, then
                      each one becomes its own cluster
                    = Clusters fit to noise or outliers

            - Clusters fit to noise or outliers
                    = K-Means may form clusters around anormalies that don't represent real structure.
            
            - Low generalization to new data:
                    = If your clusters were tuned to a specfici dataset, they may not represent new data well.
                      (e.g., future customers, different population, etc.)

    UNDERFITTING -> only one cluster.
"""
def K_means_Distance(test_idx, test_data, model):
    # Extract the feature vector of the test instance
    c1 = np.array(test_data.iloc[test_idx, :])

    # First cluster center from the model
    c2 = model.cluster_centers_[0]      # -> the first cluster center (centroid) learned by a fitted K-Means model

    temp = c1 - c2

    # Compute Euclidean distance
    euclid_dist = np.sqrt(np.dot(temp.T, temp))

    return euclid_dist

# Implementation of a K-means-based anomaly (or outlier) detection model.
def kmd_model(test_data, train_data, model, threshold_dis):
    kmd_d_raw_preds = []
    for idx in tqdm(range(len(test_data))):
        dis = K_means_Distance(idx, test_data, model)
        if dis > threshold_dis:
            kmd_d_raw_preds.append(1)
        else:
            kmd_d_raw_preds.append(0)

    return kmd_d_raw_preds

kmeans = KMeans(n_clusters=1, random_state=0).fit(d_raw_train_normal)
kmd_d_raw_preds = kmd_model( d_raw_test, d_raw_train_normal, kmeans, 0.8)
kmeans = KMeans(n_clusters=1, random_state=0).fit(d_raw_probs_train_normal)
kmd_d_raw_probs_preds = kmd_model( d_raw_probs_test, d_raw_probs_train_normal, kmeans, 0.8)
kmeans = KMeans(n_clusters=1, random_state=0).fit(d_raw_pca_train_normal)
kmd_d_raw_pca_preds = kmd_model( d_raw_pca_test, d_raw_pca_train_normal, kmeans, 0.8)
kmeans = KMeans(n_clusters=1, random_state=0).fit(d_raw_pca_probs_train_normal)
kmd_d_raw_pca_probs_preds = kmd_model( d_raw_pca_probs_test, d_raw_pca_probs_train_normal, kmeans, 0.8)
kmeans = KMeans(n_clusters=1, random_state=0).fit(d_norm_train_normal)
kmd_d_norm_preds = kmd_model( d_norm_test, d_norm_train_normal, kmeans, 0.8)
kmeans = KMeans(n_clusters=1, random_state=0).fit(d_norm_probs_train_normal)
kmd_d_norm_probs_preds = kmd_model( d_norm_probs_test, d_norm_probs_train_normal, kmeans, 0.8)
kmeans = KMeans(n_clusters=1, random_state=0).fit(d_norm_pca_train_normal)
kmd_d_norm_pca_preds = kmd_model( d_norm_pca_test, d_norm_pca_train_normal, kmeans, 0.8)
kmeans = KMeans(n_clusters=1, random_state=0).fit(d_norm_pca_probs_train_normal)
kmd_d_norm_pca_probs_preds = kmd_model( d_norm_pca_probs_test, d_norm_pca_probs_train_normal, kmeans, 0.8)

""" K-Means (standard) """
"""
    We will be applying K-Means algorithm in its standard clustering approach on each of the generated datasets.

    The classic algorithm that partitions data into k clusters by minimizing the sum of squared Euclidean distances
    between points and their assigned cluster cents.

    The above Euclidean distance is just explicitly way of stating the default distance metric (Euclidean) in K-means.
"""
def kmean_C_model(train_data, test_data):
    kmeans = KMeans(n_clusters=2).fit(train_data)
    preds = kmeans.predict(test_data)

kmean_d_raw_preds = kmean_C_model(d_raw_train, d_raw_test)
kmean_d_raw_probs_preds = kmean_C_model(d_raw_probs_train, d_raw_probs_test)
kmean_d_raw_pca_preds = kmean_C_model(d_raw_pca_train, d_raw_pca_test)
kmean_d_raw_pca_probs_preds = kmean_C_model(d_raw_pca_probs_train, d_raw_pca_probs_test)
kmean_d_norm_preds = kmean_C_model(d_norm_train, d_norm_test)
kmean_d_norm_probs_preds = kmean_C_model(d_norm_probs_train, d_norm_probs_test)
kmean_d_norm_pca_preds = kmean_C_model(d_norm_pca_train, d_norm_pca_test)
kmean_d_norm_pca_probs_preds = kmean_C_model(d_norm_pca_probs_train, d_norm_pca_probs_test)

""" EVALUATION """
"""
    E1 : d_norm
    E2 : d_norm_probs
    E3 : d_norm_pca
    E4 : d_norm_pca_probs
    E5 : d_raw
    E6 : d_raw_probs
    E7 : d_raw_pca
    E8 : d_raw_pca_probs
"""
""" Some evaluation tools """

# F1 Score
# F1 score is a metric used to evaluate the performance of a classification model, especially when the data is imbalanced
def f1(y_true, y_pred):
    return f1_score(y_true, y_pred)

# CAP
# Custom evaluation metric
#       - Reward high true positive and true negative rates (TPR and TNR)
#       - Penalize imbalance between TPR and TNR
#       - Penalize excessive positive predictions (attack percentage)
def cap(y_true, y_pred):
    tp = sum((y_true == 1) & (y_pred == 1))
    tn = sum((y_true == 0) & (y_pred == 0))
    fp = sum((y_true == 0) & (y_pred == 1))
    fn = sum((y_true == 1) & (y_pred == 0))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    if isinstance(y_pred, bool):
        attack_percentage = int(y_pred)
    else:
        attack_percentage = sum(y_pred) / len(y_pred) if len(y_pred) > 0 else 0.0

    cap_score = (tpr + tnr) * (1 - abs(tpr - tnr)) * (1 - attack_percentage)

    return cap_score

# Sensitivity
#   -> Also known as recall or true positive rate (TPR).
# High sensitivity - model is good at catching actual positives
# Low sensitivity - model is missing a lot of positives (high false negatives).
def sensitivity(y_true, y_pred):
    tp = sum((y_true == 1) & (y_pred == 1))
    fn = sum((y_true == 1) & (y_pred == 0))

    sensitivity_score = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return sensitivity_score