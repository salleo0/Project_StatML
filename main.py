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

# Redefining the lables of the input feature 'Country'
country_labels = {
    -0.09765: 'Australia',
     0.24923: 'Canada',
    -0.46841: 'New Zealand',
    -0.28519: 'Other',
     0.21128: 'Republic of Ireland',
     0.96082: 'UK',
    -0.57009: 'USA'
}

# Relative barplot of the input feature 'Country'
feat_barplot(data_frame, 'Country', country_labels)

# Redefining the lables of the input feature 'Ethnicity'
ethnicity_labels = {
    -0.50212: 'Asian',
    -1.10702: 'Black',
     1.90725: 'Mixed-Black/Asian',
     0.12600: 'Mixed-White/Asian',
    -0.22166: 'Mixed-White/Black',
     0.11440: 'Other',
    -0.31685: 'White'
}

# Relative barplot of the input feature 'Ethnicity'
feat_barplot(data_frame, 'Ethnicity', ethnicity_labels)

# Relative barplot for all the drugs
proportions_plot(data_frame, Y)

# Naive tree
naive_tree(X, y, target_column, feature_columns)

# Trees using cross validation
cv_tree(X, y, target_column, feature_columns)

# SHAP code below
shap_estimation(X, y)