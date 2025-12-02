import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score

from dataset_study import import_dataset
from decision_trees import naive_tree, cv_tree

# File name
file_path = 'drug_consumption.data'

# Create the dataframe to work with
data_frame = import_dataset(file_path)

# Select features and target drug
feature_columns = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity', 
                'Nscore', 'Escore', 'Oscore', 'Cscore', 'Ascore', 'Impulsive', 'SS']
target_column = 'Crack'

# Extract features and target
X = data_frame[feature_columns].copy()
y = data_frame[target_column].copy()

# Naive tree
naive_tree(X, y, target_column, feature_columns)

# Trees using cross validation
cv_tree(X, y, target_column, feature_columns)

# SHAP code below
