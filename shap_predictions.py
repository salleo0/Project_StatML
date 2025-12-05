import shap 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def shap_estimation(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size = 0.15,
        random_state = 42,
        shuffle = True
    )

    decision_tree = DecisionTreeClassifier(
        max_depth=4,
        random_state=42,
        class_weight='balanced'
        )
    
    decision_tree.fit(X_train, y_train)

    X_test = X_test.iloc[[0]]

    explainer = shap.TreeExplainer(decision_tree)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test)