import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score

file_path = 'drug_consumption.data'

column_names = [
    'ID', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity',
    'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS',
    'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caffeine', 'Cannabis', 'Choc', 'Coke', 'Crack',
    'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms',
    'Nicotine', 'Semeron', 'VSA'
]

drug_names = [
    'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caffeine', 'Cannabis', 'Choc', 'Coke', 'Crack',
    'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms',
    'Nicotine', 'Semeron', 'VSA'
]

data_frame = pd.read_csv(file_path, names=column_names, header=None)

# Semeron is a fictitious drug used to identify over-claimers
# We remove all instances where Semeron != CL0 and then remove the column
data_frame = data_frame[data_frame['Semeron'] == 'CL0']
data_frame.drop('Semeron', axis=1, inplace=True)
drug_names.remove('Semeron')

# We simplify the dataframe into USER (1) and NON-USER (0)
for drug in drug_names:
    data_frame[drug] = data_frame[drug].apply(lambda x: 1 if x in ['CL3','CL4','CL5','CL6'] else 0)

# Crack small dataset
# Cocaine, cannabis, LSD larger datasets

###############################################



feature_columns = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity', 
                'Nscore', 'Escore', 'Oscore', 'Cscore', 'Ascore', 'Impulsive', 'SS']
target_column = 'Cannabis'


# Extract features and target
X = data_frame[feature_columns].copy()
y = data_frame[target_column].copy()

test_size = 0.15
validation_size = 0.2
training_size = 0.65

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size = test_size,
    random_state = 42,
    shuffle = True
)


decision_tree = DecisionTreeClassifier(
    max_depth=3,
    random_state=42,
    class_weight='balanced'
    )

decision_tree.fit(X_train, y_train)

# 4. Predictions & Evaluation
y_pred = decision_tree.predict(X_test)

conf_mat = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_mat.ravel()

# Calculate metrics
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
accuracy = accuracy_score(y_test, y_pred)

print("Confusion Matrix: \n", conf_mat)
print(f"Sensitivity (Recall for Users): {sensitivity:.4f}")
print(f"Specificity (Recall for Non-Users): {specificity:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# 5. Feature Importance (Gini Importance)
importances = decision_tree.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nFeature Ranking (Gini Importance):")
for f in range(len(feature_columns)):
    # Only print features that were actually used (importance > 0)
    if importances[indices[f]] > 0:
        print(f"{feature_columns[indices[f]]}: {importances[indices[f]]:.4f}")

# 6. Plot Tree
plt.figure(figsize=(20, 10))
plot_tree(decision_tree, feature_names=feature_columns, class_names=['Non-User', 'User'], filled=True, fontsize=10)
plt.title("Decision Tree")
plt.savefig('crack_decision_tree.png')