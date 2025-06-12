# model.py
# Copyright: Shahzad Ali
# Author: Shahzad Ali
# e-mail: shahzad.ali6@unibo.it
# Created: 2025-1-24
# Last modified: 2025-06-12


import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, 
                              ExtraTreesClassifier, GradientBoostingClassifier)
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (accuracy_score, matthews_corrcoef, precision_score, recall_score, f1_score, roc_auc_score,
                             balanced_accuracy_score, confusion_matrix)
from sklearn.preprocessing import label_binarize
    
import warnings
warnings.filterwarnings('ignore')


def get_models_and_params(seed):
    """Returns a dictionary of various machine learning models initialized with a given random seed."""
    models_with_params = {
        'LR': (LogisticRegression(random_state=seed), {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2', 'elasticnet', 'none'], 'solver': ['liblinear', 'lbfgs', 'saga', 'newton-cg'], 'max_iter': [50, 100, 500, 1000, 5000]}),
        # 'LR-SGD': (SGDClassifier(loss='log_loss', random_state=seed), {'penalty': ['l2', 'l1', 'elasticnet'], 'alpha': [0.0001, 0.001, 0.01, 0.1], 'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'], 'eta0': [0.001, 0.01, 0.1, 1], 'max_iter': [100, 200, 300, 500, 1000, 1500], 'tol': [1e-2, 1e-3, 1e-4, 1e-5]}),
        # 'LDA': (LinearDiscriminantAnalysis(), {'solver': ['lsqr', 'eigen'], 'shrinkage': ['auto', None, 'log']}),
        'DT': (DecisionTreeClassifier(random_state=seed), {'criterion': ['gini', 'entropy'], 'max_depth': [3, 5, 10, 15, 20, None], 'min_samples_split': [2, 3, 5, 7, 10], 'max_features': [0.3, 0.5, 0.7, 1.0]}),
        'RF': (RandomForestClassifier(random_state=seed),{'n_estimators': [5, 10, 30, 50, 70], 'max_depth': [3, 5, 10, 15, 20, None],'min_samples_split': [2, 5, 10, 15], 'min_samples_leaf': [1, 2, 4, 6], 'bootstrap': [True, False]}),
        # 'SVM': (SVC(random_state=seed, probability=True),{'C': [0.01, 0.1, 1, 10, 100, 500, 1000], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'gamma': ['scale', 'auto']}),   # 'max_iter': [1000, 5000, 10000]
        # 'KNN': (KNeighborsClassifier(),{'n_neighbors': [1, 3, 5, 7, 9], 'weights': ['uniform', 'distance'], 'p': [1, 2], 'metric': ['euclidean', 'manhattan', 'minkowski']}),
        # 'NB': (GaussianNB(), {'var_smoothing': np.logspace(0, -9, num=100)}), 
        # 'ETC': (ExtraTreesClassifier(random_state=seed),{'n_estimators': [5, 10, 30, 50, 70], 'criterion': ['gini', 'entropy'], 'max_depth': [10, 20, 30, None],'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'max_features': [0.3, 0.5, 0.7, 1.0],'bootstrap': [True, False], 'oob_score': [True]}),
        # 'AdB': (AdaBoostClassifier(random_state=seed),{'n_estimators': [5, 10, 30, 50, 70], 'learning_rate': [0.01, 0.1, 0.5, 1.0],'algorithm': ['SAMME']}),
        # 'GB': (GradientBoostingClassifier(random_state=seed),{'n_estimators': [5, 10, 30, 50, 70], 'learning_rate': [0.01, 0.03, 0.1, 0.3, 0.5, 1.0],'max_depth': [3, 5, 7, 10], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'subsample': [0.4, 0.5, 0.6, 0.8, 1.0], 'max_features': [0.3, 0.5, 0.7, 1.0]}),
        # 'XGBC': (XGBClassifier(random_state=seed),{'n_estimators': [5, 10, 30, 50, 70], 'learning_rate': [0.01, 0.03, 0.1, 0.3, 0.5], 'max_depth': [3, 5, 7, 10],'min_child_weight': [1, 3, 5], 'subsample': [0.8, 1.0], 'colsample_bytree': [0.8, 1.0],'gamma': [0, 0.1, 0.2], 'reg_alpha': [0, 0.1, 0.5], 'reg_lambda': [1, 1.5, 2]}),
        # 'MLP': (MLPClassifier(random_state=seed),{'learning_rate_init': [0.001, 0.0005, 0.0001], 'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10], 'learning_rate': ['constant', 'invscaling', 'adaptive'],'activation': ['identity', 'logistic', 'tanh', 'relu'],'hidden_layer_sizes': [(30,), (50,), (100,), (50, 25), (100, 50), (200, 100)], 'solver': ['lbfgs', 'sgd', 'adam'], 'max_iter': [50, 100, 150, 200, 250, 300, 400, 500], 'early_stopping': [True], 'batch_size': [32, 64, 128], 'momentum': [0.5, 0.9, 0.99]}),
        }
    return models_with_params


def optimize_models(models_with_params, X_train, y_train):
    """
    Optimizes selected models with predefined parameter grids using GridSearchCV.
    
    Parameters:
        models_with_params : dict : Dictionary of models and their parameter grids.
        X_train : DataFrame : Training features.
        y_train : Series : Training target.
        
    Returns:
        optimized_models : dict : Dictionary of optimized models with their best parameters.
    """
    optimized_models = {}
    for name, (model, param_grid) in tqdm(models_with_params.items(), desc="Optimizing Models", unit="model"):
        if param_grid:  # Check if the parameter grid is not empty
            print(f"Optimizing {name} with GridSearchCV...")
            grid_search = GridSearchCV(model, param_grid, cv=5, verbose=1, n_jobs=-1, scoring='accuracy') #n_iter=20, n_jobs=-1 for all available cpus
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            optimized_models[name] = best_model

            print(f"Best parameters for {name}: {best_params}")
        else:
            print(f"No parameter grid for {name}; using default model.")
            optimized_models[name] = model  # Use the default model if no parameters to optimize

    return optimized_models


def evaluate_models(models, X_test, y_test):
    """Evaluates each trained model on the test data and returns performance metrics and predictions."""
    eval_metrics = []
    model_names = []
    predictions = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        predictions[name] = y_pred

        accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
        bacc = round(balanced_accuracy_score(y_test, y_pred) * 100, 2)
        mcc = round(matthews_corrcoef(y_test, y_pred) * 100, 2)
        precision = round(precision_score(y_test, y_pred, average='weighted') * 100, 2)
        recall = round(recall_score(y_test, y_pred, average='weighted') * 100, 2)
        fscore = round(f1_score(y_test, y_pred, average='weighted') * 100, 2)
        

        # Confusion matrix and error calculations
        cm = confusion_matrix(y_test, y_pred)
        
        if cm.shape == (2, 2):  # Binary classification
            tn, fp, fn, tp = cm.ravel()

            specificity = round((tn / (tn + fp)) * 100, 2)  # True Negative Rate
            type_i_error = round(fp / (fp + tn) * 100, 2)  # False Positive Rate
            type_ii_error = round(fn / (tp + fn) * 100, 2)  # False Negative Rate
            #ppv = precision  # PPV is equivalent to precision in binary classification
            npv = round((tn / (tn + fn)) * 100, 2)   # Negative Predictive Value (NPV)

        else:  # Multi-class classification
            # Calculate Type I and Type II errors for each class
            fp = cm.sum(axis=0) - np.diag(cm)  # False positives per class
            fn = cm.sum(axis=1) - np.diag(cm)  # False negatives per class
            tp = np.diag(cm)  # True positives per class
            tn = cm.sum() - (fp + fn + tp)  # True negatives per class
            
            # Mean calculations for multi-class
            specificity = round((tn / (tn + fp)).mean() * 100, 2) # Mean True Negative Rate
            type_i_error = round((fp / (fp + tn)).mean() * 100, 2)  # Mean False Positive Rate
            type_ii_error = round((fn / (fn + tp)).mean() * 100, 2)  # Mean False Negative Rate
            #ppv = precision  # Mean precision for weighted calculation
            npv = round((tn / (tn + fn)).mean() * 100, 2)

        # Calculate AUC (if applicable)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            if len(set(y_test)) > 2:  # Multiclass case
                y_test_bin = label_binarize(y_test, classes=sorted(set(y_test)))
                auc_score = round(roc_auc_score(y_test_bin, y_prob, average='weighted', multi_class='ovr') * 100, 2)
            else:  # Binary case
                auc_score = round(roc_auc_score(y_test, y_prob[:, 1]) * 100, 2)
        else:
            auc_score = 'N/A'

        eval_metrics.append([accuracy, bacc, mcc, precision, recall, specificity, fscore, auc_score, npv, type_i_error, type_ii_error])
        model_names.append(name)
    
    metrics = pd.DataFrame(eval_metrics, columns=['Accuracy', 'BACC', 'MCC', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'AUC', 'NPV', 'Type I Error', 'Type II Error'], index=model_names)
    return metrics, predictions

def select_top_models(metrics, models, top_n=3):
    """Select the top N models based on accuracy or another metric."""
    # Sort by the chosen metric (e.g., "Accuracy")
    top_models_df = metrics.sort_values(by='Accuracy', ascending=False).head(top_n)

    # Extract the model names from the "Model" column in the sorted DataFrame
    top_model_names = top_models_df['Model'].tolist()

    # Retrieve the actual model objects from the models dictionary
    return [(name, models[name]) for name in top_model_names if name in models]

def select_top_model_1(metrics, models, top_n=3):
    """
    Select the top N models based on their evaluation metrics.
    Parameters:
        metrics : DataFrame : The evaluation metrics for the models.
        models : dict : Dictionary of trained models.
        top_n : int : The number of top models to select.
    Returns:
        top_models : list : List of the top N models with their names.
    """
    top_model_names = metrics.nlargest(top_n, 'Accuracy').index.tolist()
    return [(name, models[name]) for name in top_model_names]


################################
def calculate_q_metric(predictions_1, predictions_2, y_true):
    """Calculate the Q metric to assess diversity between two classifiers."""
    N11 = np.sum((predictions_1 == y_true) & (predictions_2 == y_true))
    N00 = np.sum((predictions_1 != y_true) & (predictions_2 != y_true))
    N01 = np.sum((predictions_1 != y_true) & (predictions_2 == y_true))
    N10 = np.sum((predictions_1 == y_true) & (predictions_2 != y_true))
    
    numerator = N11 * N00 - N01 * N10
    denominator = N11 * N00 + N01 * N10
    
    if denominator == 0:
        return np.inf  # Avoid division by zero
    
    return numerator / denominator


def select_diverse_models(metrics, predictions, y_true, top_n=5):
    """
    Selects pairs of models with low Q values to form a diverse ensemble.
    Returns a list of model names based on the Q metric.
    """
    # Filter out ensemble models if present in metrics or predictions
    ensemble_models = ['Ensemble_hV', 'Ensemble_sV']
    model_names = [name for name in metrics.index if name not in ensemble_models]

    pairs = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            model1, model2 = model_names[i], model_names[j]
            q_value = calculate_q_metric(predictions[model1], predictions[model2], y_true)
            pairs.append((q_value, model1, model2))
    
    # Sort pairs by Q value (ascending) and select the most diverse models
    pairs = sorted(pairs, key=lambda x: x[0])
    selected_models = set()
    for _, name1, name2 in pairs[:top_n]:  # Select top N diverse pairs
        selected_models.add(name1)
        selected_models.add(name2)
    
    return list(selected_models)

