import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score


def naive_tree(X, y, target_column, feature_columns):
    # Approach 1: naive decision tree
    print("Naive tree")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size = 0.15,
        random_state = 42,
        shuffle = True
    )

    # Train the model
    decision_tree = DecisionTreeClassifier(
        max_depth=3,
        random_state=42,
        class_weight='balanced'
        )
    decision_tree.fit(X_train, y_train)

    # Get predicted values
    y_pred = decision_tree.predict(X_test)

    # Compute confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_mat.ravel()

    # Compute metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTarget: {target_column}")
    print("Confusion Matrix: \n", conf_mat)
    print(f"Sensitivity (Recall for Users): {sensitivity:.4f}")
    print(f"Specificity (Recall for Non-Users): {specificity:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    # Feature Importance (Gini Importance)
    importances = decision_tree.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\nFeature Ranking (Gini Importance):")
    for f in range(len(feature_columns)):
        # Only print features that were actually used (importance > 0)
        if importances[indices[f]] > 0:
            print(f"{feature_columns[indices[f]]}: {importances[indices[f]]:.4f}")

    # Plot Tree
    plt.figure(figsize=(20, 10))
    plot_tree(decision_tree, feature_names=feature_columns, class_names=['Non-User', 'User'], filled=True, fontsize=10)
    plt.title("Decision Tree")
    plt.savefig('Pictures\\decision_tree.png')

    print("_" * 46)


def cv_tree(X,y, target_column, feature_columns):

    # Approach 2: using K-fold cross validation
    print("Trees with Cross Validation")

    # Setup Stratified K-Fold (5 splits is standard)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Lists to store results from each fold
    accuracies = []
    sensitivities = []
    specificities = []

    print(f"\nTarget: {target_column}")
    print(f"{'Fold':<5} | {'Accuracy':<10} | {'Sensitivity':<12} | {'Specificity':<12}")
    print("-" * 46)

    fold_num = 1

    # Perform multiple splits and create the trees
    for train_index, test_index in skf.split(X, y):

        # Split Data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Train the model
        decision_tree = DecisionTreeClassifier(
            max_depth=3,           # Keep tree simple
            class_weight='balanced',
            random_state=42
        )
        decision_tree.fit(X_train, y_train)
        
        # Get predicted values
        y_pred = decision_tree.predict(X_test)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Compute metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = accuracy_score(y_test, y_pred)

        # Save results
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        accuracies.append(accuracy)

        print(f"{fold_num:<5} | {accuracy:.4f}     | {sensitivity:.4f}       | {specificity:.4f}")
        fold_num += 1

    # Final Average Results
    print("-" * 46)
    print(f"Mean Accuracy:    {np.mean(accuracies):.4f}")
    print(f"Mean Sensitivity: {np.mean(sensitivities):.4f} (robust score)")
    print(f"Mean Specificity: {np.mean(specificities):.4f}")

    # Plot the tree from the last fold (Visual Aid)
    plt.figure(figsize=(20, 10))
    plot_tree(decision_tree, feature_names=feature_columns, class_names=['Non-User', 'User'], filled=True, fontsize=10)
    plt.title(f"Decision Tree for {target_column} (Fold {fold_num-1})")
    plt.savefig('Pictures\\decision_tree_fold5.png')