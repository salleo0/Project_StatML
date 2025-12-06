from dataset_study import import_dataset, feat_barplot, proportions_plot
from decision_trees import naive_tree, cv_tree
from shap_predictions import shap_estimation

# File name
file_path = 'drug_consumption.data'

# Create the dataframe to work with and analyze the dataset
data_frame = import_dataset(file_path)

# Select features and target drugs
# Note that Country and Ethnicity are not considered
feature_columns = ['Age', 'Gender', 'Education', 
                   'Nscore', 'Escore', 'Oscore', 'Cscore', 'Ascore', 'Impulsive', 'SS']
target_column = 'LSD'

# Output features 
output_columns = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caffeine', 'Cannabis',
                  'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine',
                  'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'VSA']

# Extract input features, output features and target
X = data_frame[feature_columns].copy()
Y = data_frame[output_columns].copy()
y = data_frame[target_column].copy()

# Relative barplot of the input feature 'Country'
feat_barplot(data_frame, 'Country')

#Relative barplot of the input feature 'Ethnicity'
feat_barplot(data_frame, 'Ethnicity')

# Relative barplot for all the drugs
proportions_plot(data_frame, Y)

# Naive tree
naive_tree(X, y, target_column, feature_columns)

# Trees using cross validation
cv_tree(X, y, target_column, feature_columns)

# SHAP code below
shap_estimation(X, y)