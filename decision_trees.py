import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score


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
        max_depth=4,
        random_state=42,
        #class_weight='balanced'
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
    precision = tp / (fp + tp) if (fp + tp) > 0 else 0

    print(f"\nTarget: {target_column}")
    print("Confusion Matrix: \n", conf_mat)
    print(f"Sensitivity (Recall for Users): {sensitivity:.4f}")
    print(f"Specificity (Recall for Non-Users): {specificity:.4f}")
    print(f"Precision: {precision:.4f}")

    # ROC curve
    # Get probabilities for the positive class (index 1)
    y_probs = decision_tree.predict_proba(X_test)[:, 1]

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    # Plot the curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig('Pictures/roc_curve_naive.png')

    # Calculate Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_probs)
    avg_precision = average_precision_score(y_test, y_probs)
    baseline = len(y_test[y_test==1]) / len(y_test) # Ratio of positives

    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, color='green', lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
    plt.plot([0, 1], [baseline, baseline], linestyle='--', color='navy', label=f'Baseline ({baseline:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Naive)')
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.savefig('Pictures/pr_curve_naive.png')
    plt.close()

    
    # Get Gini importances
    importances = decision_tree.feature_importances_
    # Sort their indices from most to least important
    indices = np.argsort(importances)[::-1]
    # Print them in order
    print("\nFeature Ranking (Gini Importance):")
    for f in range(len(feature_columns)):
        # Only print features that were actually used (importance > 0)
        if importances[indices[f]] > 0:
            print(f"{feature_columns[indices[f]]}: {importances[indices[f]]:.4f}")

    # Plot Tree
    plt.figure(figsize=(20, 10))
    plot_tree(decision_tree, feature_names=feature_columns, class_names=['Non-User', 'User'], filled=True, fontsize=10)
    plt.title("Decision Tree")
    plt.savefig('Pictures\\decision_tree_naive.png')

    print("_" * 46)


def cv_tree(X,y, target_column, feature_columns):
    # Approach 2: using K-fold cross validation
    print("Trees with Cross Validation")

    # Setup Stratified K-Fold (5 splits is standard)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Initializations
    # For plotting ROC curves
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100) # Common X-axis for interpolation
    plt.figure(figsize=(10, 8))
    
    # Lists to store results from each fold
    precisions = []
    sensitivities = []
    specificities = []
    
    precisions_interp = []
    aps = [] # Average Precision Scores
    mean_recall = np.linspace(0, 1, 100) # Common X-axis for PR interpolation

    print(f"\nTarget: {target_column}")
    print(f"{'Fold':<5} | {'Precision':<10} | {'Sensitivity':<12} | {'Specificity':<12}")
    print("-" * 46)

    fold_num = 1

    # Perform multiple splits and create the trees
    for train_index, test_index in skf.split(X, y):

        # Split Data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Train the model
        decision_tree = DecisionTreeClassifier(
            max_depth=3,
            class_weight='balanced',
            random_state=42
        )
        decision_tree.fit(X_train, y_train)
        
        # Get Probabilities for ROC
        y_probs = decision_tree.predict_proba(X_test)[:, 1]
        
        # Compute ROC for this fold
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # Interpolate TPR
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        
        # Plot individual fold curve
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC Fold {fold_num} (AUC = {roc_auc:.2f})')

        # Calculate precision-recall curve
        p_fold, r_fold, _ = precision_recall_curve(y_test, y_probs)
        ap_fold = average_precision_score(y_test, y_probs)
        aps.append(ap_fold)
        
        # Interpolate Precision (Note: r_fold is usually descending, so we reverse for interp)
        interp_p = np.interp(mean_recall, r_fold[::-1], p_fold[::-1])
        precisions_interp.append(interp_p)

        # Get predicted values
        y_pred = decision_tree.predict(X_test)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Compute metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (fp + tp) if (fp + tp) > 0 else 0

        # Save results
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        precisions.append(precision)

        print(f"{fold_num:<5} | {precision:.4f}     | {sensitivity:.4f}       | {specificity:.4f}")
        fold_num += 1


    # Plot the average ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0 # Ensure curve ends at 1
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)


    plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})', lw=2, alpha=.8)
    # Plot Random Guess Line
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random Guess', alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Cross-Validated ROC for {target_column}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig('Pictures/roc_curve_cv.png')
    plt.close()

    plt.figure(figsize=(10, 8))
    
    mean_precision = np.mean(precisions_interp, axis=0)
    mean_ap = np.mean(aps)
    std_ap = np.std(aps)
    
    # Calculate baseline (fraction of positives in dataset)
    baseline = len(y[y==1]) / len(y)

    plt.plot(mean_recall, mean_precision, color='green', label=f'Mean PR (AP = {mean_ap:.2f} $\pm$ {std_ap:.2f})', lw=2, alpha=.8)
    plt.plot([0, 1], [baseline, baseline], linestyle='--', lw=2, color='navy', label=f'Baseline ({baseline:.2f})', alpha=.8)
    
    # Add standard deviation area
    std_precision = np.std(precisions_interp, axis=0)
    p_upper = np.minimum(mean_precision + std_precision, 1)
    p_lower = np.maximum(mean_precision - std_precision, 0)
    plt.fill_between(mean_recall, p_lower, p_upper, color='lightgreen', alpha=.3, label=r'$\pm$ 1 std. dev.')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Cross-Validated Precision-Recall Curve for {target_column}')
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.savefig('Pictures/pr_curve_cv.png')
    plt.close()

    # Final Average Results
    print("-" * 46)
    print(f"Mean Precision:    {np.mean(precisions):.4f}")
    print(f"Mean Sensitivity: {np.mean(sensitivities):.4f} (robust score)")
    print(f"Mean Specificity: {np.mean(specificities):.4f}")
    print(f"Mean AUC: {mean_auc:.4f}")

    # Plot the tree from the last fold (Visual Aid)
    plt.figure(figsize=(20, 10))
    plot_tree(decision_tree, feature_names=feature_columns, class_names=['Non-User', 'User'], filled=True, fontsize=10)
    plt.title(f"Decision Tree for {target_column} (Fold {fold_num-1})")
    plt.savefig('Pictures\\decision_tree_fold5.png')
    plt.close()
    
   