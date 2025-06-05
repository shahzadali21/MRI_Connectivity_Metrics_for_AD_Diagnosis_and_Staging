# regression_models.py

import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct

from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, explained_variance_score, mean_absolute_percentage_error, max_error)
from math import log

import warnings
warnings.filterwarnings('ignore')


def get_models_and_params(seed): 
    return {
        # 'LR': (LinearRegression(), {'fit_intercept': [True, False]}),
        # 'Ridge': (Ridge(random_state=seed), {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}),
        # 'Lasso': (Lasso(random_state=seed), {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}),
        # 'ElasticNet': (ElasticNet(random_state=seed), {'alpha': [0.001, 0.01, 0.1, 1], 'l1_ratio': [0.1, 0.5, 0.7, 1.0]}),
        'DT': (DecisionTreeRegressor(random_state=seed), {'max_depth': [3, 5, 10, 15, 20, None], 'min_samples_split': [2, 5, 10], 'max_features': [0.3, 0.5, 0.7, 1.0]}),
        'RF': (RandomForestRegressor(random_state=seed), {'n_estimators': [10, 50, 100], 'max_depth': [3, 5, 10, 15, 20, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'bootstrap': [True, False]}),
        # 'SVR': (SVR(), {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'epsilon': [0.01, 0.1, 1]}),
        # 'GPR': (GaussianProcessRegressor(), {'kernel': [1.0 * RBF(length_scale=1.0), 1.0 * Matern(length_scale=1.0, nu=1.5), 1.0 * RationalQuadratic(length_scale=1.0, alpha=1.0), 1.0 * DotProduct(sigma_0=1.0)], 'alpha': [1e-10, 1e-5, 1e-2], 'n_restarts_optimizer': [0, 5, 10]}),
        'AdaB': (AdaBoostRegressor(random_state=seed), {'n_estimators': [10, 50, 100], 'learning_rate': [0.01, 0.1, 0.5, 1.0]}),
        # 'GB': (GradientBoostingRegressor(random_state=seed), {'n_estimators': [10, 50, 100], 'learning_rate': [0.01, 0.1, 0.3], 'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'subsample': [0.5, 0.7, 1.0], 'max_features': [0.3, 0.5, 0.7, 1.0]}),
        # 'XGB': (XGBRegressor(random_state=seed), {'n_estimators': [10, 50, 100], 'learning_rate': [0.01, 0.1, 0.3], 'max_depth': [3, 5, 7], 'min_child_weight': [1, 3, 5], 'subsample': [0.8, 1.0], 'colsample_bytree': [0.8, 1.0], 'gamma': [0, 0.1, 0.2], 'reg_alpha': [0, 0.1, 0.5], 'reg_lambda': [1, 1.5, 2]}),
        # 'Bagging': (BaggingRegressor(random_state=seed), {'n_estimators': [10, 50, 100], 'max_samples': [0.3, 0.5, 0.7, 1.0], 'max_features': [0.3, 0.5, 0.7, 1.0]})
    }


def optimize_models(models_with_params, X_train, y_train):
    """
    Optimizes selected regression models with predefined parameter grids using GridSearchCV.
    """
    optimized_models = {}
    for name, (model, param_grid) in tqdm(models_with_params.items(), desc="Optimizing Models", unit="model"):
        if param_grid:  # Check if the parameter grid is not empty
            print(f"Optimizing {name} with GridSearchCV...")
            grid_search = GridSearchCV(model, param_grid, cv=5, verbose=1, n_jobs=-1, scoring='neg_mean_squared_error')
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
    """
    Evaluates each trained model on the test data and returns performance metrics and predictions.
    """
    eval_metrics = []
    model_names = []
    predictions = {}

    n = len(y_test)  # Number of samples in the test set
    mean_y_test = np.mean(y_test)  # Mean of the true values for NSE calculation

    for name, model in models.items():
        y_pred = model.predict(X_test)
        predictions[name] = y_pred

        mae = round(mean_absolute_error(y_test, y_pred), 2)
        mse = round(mean_squared_error(y_test, y_pred), 2)
        rmse = round(np.sqrt(mse), 2)
        r2 = round(r2_score(y_test, y_pred), 2)
        evs = round(explained_variance_score(y_test, y_pred), 2)
        #mape = round(mean_absolute_percentage_error(y_test, y_pred), 2)
        max_err = round(max_error(y_test, y_pred), 2)

        # Residual Sum of Squares (RSS)
        rss = np.sum((y_test - y_pred) ** 2)

        # Bayesian Information Criterion (BIC)
        p = X_test.shape[1] + 1  # Number of features + intercept
        bic = round(n * np.log(rss / n) + p * np.log(n), 2)

        # Akaike Information Criterion (AIC)
        aic = round(n * np.log(rss / n) + 2 * p, 2)

        # Maximum Likelihood (approximation)
        sigma_squared = rss / n  # Variance of residuals
        log_likelihood = -n / 2 * np.log(2 * np.pi * sigma_squared) - rss / (2 * sigma_squared)
        max_likelihood = round(log_likelihood, 2)

        # Nash-Sutcliffe Efficiency (NSE)
        numerator = np.sum((y_test - y_pred) ** 2)
        denominator = np.sum((y_test - mean_y_test) ** 2)
        nse = round(1 - (numerator / denominator), 2)

        # Adjusted R²
        adjusted_r2 = round(1 - ((1 - r2) * (n - 1)) / (n - p - 1), 2)

        eval_metrics.append([mae, mse, rmse, r2, adjusted_r2, evs, max_err, rss, bic, aic, max_likelihood, nse])
        model_names.append(name)

    metrics = pd.DataFrame(eval_metrics, columns=['MAE', 'MSE', 'RMSE', 'R2', 'Adjusted R2', 'Explained Variance', 'Max Error', 'RSS', 'BIC', 'AIC', 'Max Likelihood', 'NSE'], index=model_names)
    return metrics, predictions


def select_top_models(eval_metrics, models, top_n=3):
    """    Selects the top N models based on R² from the evaluation metrics    """
    if 'R2' not in eval_metrics.columns:
        raise ValueError("The evaluation metrics DataFrame must contain an 'R2' column.")

    # Sort metrics by R² in descending order
    top_model_names = eval_metrics.sort_values(by='R2', ascending=False).head(top_n)['Model'].tolist()

    # Select models from the dictionary based on their names
    top_models = [(name, models[name]) for name in top_model_names if name in models]

    if len(top_models) < top_n:
        logging.warning(f"Only {len(top_models)} out of {top_n} requested models were found.")

    return top_models
