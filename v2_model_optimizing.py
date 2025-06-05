# Copyright: Shahzad Ali
# Author: Shahzad Ali
# e-mail: shahzad.ali6@unibo.it
# Created: 2025-1-24
# Last modified: 2025-03-07

"""
This script handles the training, optimization, and ensemble of machine learning models.
It loads preprocessed data, performs grid search optimization, evaluates the models,
calculates diversity using the Q metric, and performs ensemble voting with diverse models.
"""

# Before using this script, run `preprocessing_v2.py`.

import os
import logging
import argparse

from utils import load_data, save_model, load_model, save_results, save_metrics_to_excel
from V2_models import get_models_and_params, optimize_models, evaluate_models


import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

# Create directory structure for feature combinations and comparisons
def create_directory_structure(output_dir, feature_combination_name, classification_type, comparison):
    feature_combination_dir = os.path.join(output_dir, feature_combination_name)
    os.makedirs(feature_combination_dir, exist_ok=True)

    # Create subdirectory for each classification comparison within the feature combination directory
    comparison_folder_name = comparison.replace('vs_', '') if classification_type == 'binary' else 'CN_MCI_AD'
    classification_dir = os.path.join(feature_combination_dir, comparison_folder_name)
    os.makedirs(os.path.join(classification_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(classification_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(classification_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(classification_dir, 'results'), exist_ok=True)

    return classification_dir


def train_and_evaluate_models(data_dir, models_dir, results_dir, classification_type, comparison, metrics_dict):
    logging.info("Loading preprocessed data")
    X_train = load_data(os.path.join(data_dir, 'X_train.csv'))
    X_test = load_data(os.path.join(data_dir, 'X_test.csv'))
    y_train = load_data(os.path.join(data_dir, 'y_train.csv')).values.ravel()
    y_test = load_data(os.path.join(data_dir, 'y_test.csv')).values.ravel()

    # Step 1: Get initial models and parameters
    logging.info("Getting models from 'models.py'")
    models_with_params = get_models_and_params(seed=42)
    optimized_models = {}


    # Step 2: Check if models are already saved
    for name, _ in models_with_params.items():
        model_path = os.path.join(models_dir, f"{name}.joblib")  # Path to the model
        if os.path.exists(model_path):
            logging.info(f"Model {name} already exists. Loading the model.")
            optimized_models[name] = load_model(name, models_dir)  # Pass model name and directory
        else:
            logging.info(f"Training and Optimizing model: {name}")
            optimized_models[name] = optimize_models({name: models_with_params[name]}, X_train, y_train)[name]
            save_model(optimized_models[name], name, models_dir)  # Ensure models are saved as .joblib

    # Step 2: Optimize models
    #logging.info(f"Training and Optimizing models")
    #optimized_models = optimize_models(models_with_params, X_train, y_train)

    #logging.info("Saving trained and Optimized models")
    #for name, model in optimized_models.items():
        #save_model(model, name, models_dir)

    # Step 3: Evaluate models
    logging.info("Evaluating models and getting predictions")
    metrics, predictions = evaluate_models(optimized_models, X_test, y_test)

    logging.info("Saving metrics for all models")
    metrics_dict[f"{classification_type}_{comparison}"] = metrics
    print("Model Performance Evaluation Metrics:")
    print(metrics.sort_values(by='Accuracy', ascending=False))

    # Combine all predictions, including those from the ensemble models
    save_results(y_test, predictions, os.path.join(results_dir, 'predictions.csv'))
    

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train and evaluate models for all classification types")
    parser.add_argument('--output_dir', type=str, default='V6_ProjectOutput_AmyStatus', help="Main project directory for clinical AD dataset")
    #parser.add_argument('--feature_combination_name', type=str, required=True, help="Name of the feature combination being processed")
    return parser.parse_args()

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = parse_arguments()

    # Dictionary to hold metrics for each classification type
    metrics_dict = {}

    # Loop over all feature combination folders
    #feature_combination_folders = [folder for folder in os.listdir(args.output_dir) if os.path.isdir(os.path.join(args.output_dir, folder))]
    # OR 
    # Loop over specified feature combination folders
    feature_combinations = [
                            #'Clinical',
                            'Morphometric',
                            'Microstructural',
                            'GT_Local',
                            'GT_Global',
                            'GT',
                            'Microstructural_Morphometric',
                            'Morphometric_GT',
                            'Microstructural_GT',
                            'Microstructural_Morphometric_GT',
                            #'Demographic_Microstructural_GT',
                            # 'Demographic_Microstructural_Morphometric_GT',
                            #'GT_Microstructural_Morphometric_Age',
                            #'GT_Microstructural_Morphometric_Sex',
                            #'GT_Microstructural_Morphometric_Edu',
                            #'GT_Microstructural_Morphometric_Age_Sex',
                            ]
    
    # Define classification types and comparisons
    CLASSIFICATION_COMPARISONS = {
        'binary': ['CN_vs_AD', 'CN_vs_MCI', 'MCI_vs_AD'],
        'three_level': ['CN_MCI_AD']
    }
    
    #for feature_combination_name in feature_combination_folders:
    for feature_combination_name in feature_combinations:
        # Loop over all classification types and comparisons
        for classification_type, comparisons in CLASSIFICATION_COMPARISONS.items():
            for comparison in comparisons:
                logging.info(f"###############################################################")
                logging.info(f"Processing {classification_type} classification: {comparison}")
                logging.info(f"###############################################################")

                # Create directory structure for the current classification and comparison
                classification_dir = create_directory_structure(
                    output_dir=args.output_dir,
                    feature_combination_name=feature_combination_name,
                    classification_type=classification_type,
                    comparison=comparison
                )
                data_dir = os.path.join(classification_dir, 'data')
                models_dir = os.path.join(classification_dir, 'models')
                results_dir = os.path.join(classification_dir, 'results')

                # Train and evaluate models for the current comparison
                train_and_evaluate_models(data_dir, models_dir, results_dir, classification_type, comparison, metrics_dict)


       # Save all metrics to a single Excel file in the main folder with unique names
        output_file = os.path.join(args.output_dir, f'ClassificationMetrics_{feature_combination_name}.xlsx')
        save_metrics_to_excel(metrics_dict, output_file)

if __name__ == "__main__":
    main()
