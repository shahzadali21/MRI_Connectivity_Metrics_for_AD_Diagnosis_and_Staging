# regression_model_optimizing.py

# Copyright: Shahzad Ali
# Author: Shahzad Ali
# e-mail: shahzad.ali6@unibo.it
# Created: 2024-10-24
# Last modified: 2024-11-16

"""
This script handles the training, optimization, and ensemble of regression models.
It includes feature selection and calculates metrics like MAE, MSE, RMSE, and R².
"""

import os
import logging
import argparse
import pandas as pd
from utils import load_data, save_model, load_model, save_results, save_metrics_to_excel  #save_regression_metrics
from v3_models import get_models_and_params, optimize_models, evaluate_models
from v3fs_feature_selection import feature_selection

import warnings
warnings.filterwarnings("ignore")

# Create directory structure for feature combinations and subsets
def create_directory_structure(output_dir, feature_combination_name, subset_name):
    feature_combination_dir = os.path.join(output_dir, feature_combination_name)
    os.makedirs(feature_combination_dir, exist_ok=True)

    # Create subdirectory for each subset within the feature combination directory
    subset_dir = os.path.join(feature_combination_dir, subset_name)
    os.makedirs(os.path.join(subset_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(subset_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(subset_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(subset_dir, 'results'), exist_ok=True)

    return subset_dir

def train_and_evaluate_models(data_dir, models_dir, results_dir, subset_name, metrics_dict, feature_selection_method=None, k=None):
    logging.info("Loading preprocessed data")
    X_train = load_data(os.path.join(data_dir, 'X_train.csv'))
    X_test = load_data(os.path.join(data_dir, 'X_test.csv'))
    y_train = load_data(os.path.join(data_dir, 'y_train.csv')).values.ravel()
    y_test = load_data(os.path.join(data_dir, 'y_test.csv')).values.ravel()

    # Step 1: Feature Selection
    if feature_selection_method:
        logging.info(f"Performing feature selection using: {feature_selection_method}")
        X_train_fs, selected_features = feature_selection(X_train, y_train, method=feature_selection_method, k=k)
        # Apply the same feature selection to the test set
        X_test_fs = X_test.iloc[:, selected_features] if isinstance(X_test, pd.DataFrame) else X_test[:, selected_features]
    else:
        logging.info("No feature selection applied.")
        X_train_fs = X_train
        X_test_fs = X_test

    # Step 2: Get initial models and parameters
    logging.info("Getting models from 'regression_models.py'")
    models_with_params = get_models_and_params(seed=42)
    optimized_models = {}

    # Step 3: Train and optimize models, also Check if models are already saved
    for name, _ in models_with_params.items():
        model_path = os.path.join(models_dir, f"{name}.joblib")
        if os.path.exists(model_path):
            logging.info(f"Model {name} already exists. Loading the model.")
            optimized_models[name] = load_model(name, models_dir)
        else:
            logging.info(f"Training and Optimizing model: {name}")
            optimized_models[name] = optimize_models({name: models_with_params[name]}, X_train_fs, y_train)[name]
            save_model(optimized_models[name], name, models_dir)

    # Step 4: Evaluate models
    logging.info("Evaluating models and getting predictions")
    metrics, predictions = evaluate_models(optimized_models, X_test_fs, y_test)

    logging.info("Saving metrics for all models")
    metrics_dict[f"{subset_name}"] = metrics
    print("Model Performance Evaluation Metrics:")
    print(metrics.sort_values(by='R2', ascending=False))

    # Save predictions
    save_results(y_test, predictions, os.path.join(results_dir, 'predictions.csv'))

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train and evaluate regression models with feature selection")
    parser.add_argument('--output_dir', type=str, default='V3fs_ProjectOutput', help="Main project directory")
    parser.add_argument('--feature_selection_method', type=str, default='pca', choices=["none", "mutual_info", "anova", "rfe_elastic_net", "random_forest", "pca", "ga"], help="Feature selection method")
    parser.add_argument('--k', type=int, default=10, help="Number of features to select")
    return parser.parse_args()

3
def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = parse_arguments()

    # Dictionary to hold metrics for each subset
    metrics_dict = {}

    # Define feature combinations
    feature_combinations = ["Demographic_Microstructural_Morphometric_GT"]

    # Subsets for regression
    SUBSETS = ["without_nan"]   #"without_nan_zero"

    for feature_combination_name in feature_combinations:
        # Loop over regression subsets
        for subset_name in SUBSETS:
            logging.info(f"###############################################################")
            logging.info(f"Processing subset: {subset_name}")
            logging.info(f"###############################################################")

            # Create directory structure for the current subset
            subset_dir = create_directory_structure(
                output_dir=args.output_dir,
                feature_combination_name=feature_combination_name,
                subset_name=subset_name
            )
            data_dir = os.path.join(subset_dir, 'data')
            models_dir = os.path.join(subset_dir, 'models')
            results_dir = os.path.join(subset_dir, 'results')

            # Train and evaluate models
            train_and_evaluate_models(
                data_dir, models_dir, results_dir, subset_name,
                metrics_dict, args.feature_selection_method, args.k
            )
        # Save metrics to a single Excel file for each feature combination
        output_file = os.path.join(args.output_dir, f'RegressionMetrics_{feature_combination_name}.xlsx')
        save_metrics_to_excel(metrics_dict, output_file)


if __name__ == "__main__":
    main()
