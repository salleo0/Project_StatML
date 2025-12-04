from dataset_study import import_dataset
from decision_trees import naive_tree, cv_tree

# File name
file_path = 'drug_consumption.data'

# Create the dataframe to work with and analyze the dataset
data_frame = import_dataset(file_path)

# Select features and target drugs
# Note that Country and Ethnicity are not considered
feature_columns = ['Age', 'Gender', 'Education', 
                'Nscore', 'Escore', 'Oscore', 'Cscore', 'Ascore', 'Impulsive', 'SS']
target_column = 'Cannabis'

# Extract features and target
X = data_frame[feature_columns].copy()
y = data_frame[target_column].copy()

# Naive tree
naive_tree(X, y, target_column, feature_columns)

# Trees using cross validation
cv_tree(X, y, target_column, feature_columns)

# SHAP code below
