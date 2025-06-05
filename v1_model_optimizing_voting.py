# Copyright: Shahzad Ali
# Author: Shahzad Ali
# e-mail: shahzad.ali6@unibo.it
# Created: 2024-10-24
# Last modified: 2024-11-07

"""
This script handles the training, optimization, and ensemble of machine learning models.
It loads preprocessed data, performs grid search optimization, evaluates the models,
calculates diversity using the Q metric, and performs ensemble voting with diverse models.
"""

# Before using this script, run `preprocessing.py`.

import os
import logging
import argparse
import pandas as pd
from sklearn.ensemble import VotingClassifier

from utils import load_data, save_model, save_results, save_metrics_to_excel, create_directory_structure
from models import get_models_and_params, optimize_models, evaluate_models, select_diverse_models


import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")


# Define classification types and comparisons
CLASSIFICATION_COMPARISONS = {
    'binary': ['CN_vs_AD', 'CN_vs_MCI', 'MCI_vs_AD'],
    'three_level': ['CN_MCI_AD']
}


def train_and_evaluate_models(data_dir, models_dir, results_dir, classification_type, comparison, metrics_dict, selection_method='diverse'):
    logging.info("Loading preprocessed data")
    X_train = load_data(os.path.join(data_dir, 'X_train.csv'))
    X_test = load_data(os.path.join(data_dir, 'X_test.csv'))
    y_train = load_data(os.path.join(data_dir, 'y_train.csv')).values.ravel()
    y_test = load_data(os.path.join(data_dir, 'y_test.csv')).values.ravel()

    # Step 1: Get initial models and parameters
    logging.info("Getting models from 'models.py'")
    models_with_params = get_models_and_params(seed=42)

    # Step 2: Optimize models
    logging.info("Training and Optimizing models")
    optimized_models = optimize_models(models_with_params, X_train, y_train)

    logging.info("Saving trained and Optimized models")
    for name, model in optimized_models.items():
        save_model(model, name, models_dir)

    # Step 3: Evaluate models
    logging.info("Evaluating models and getting predictions")
    metrics, predictions = evaluate_models(optimized_models, X_test, y_test)


    # Step 4: Select models based on the specified selection method (diverse or top)
    selected_model_names = []

    if selection_method == 'diverse':
        # Option 1: Select diverse models based on Q metric
        logging.info("Selecting diverse models based on Q metric")
        selected_model_names = select_diverse_models(metrics, predictions, y_test, top_n=3)

    elif selection_method == 'above_mean_accuracy':
        # Option 2: Select models with accuracy above the mean accuracy
        mean_accuracy = metrics['Accuracy'].mean()
        logging.info(f"Mean model accuracy: {mean_accuracy}")
        selected_model_names = metrics[metrics['Accuracy'] > mean_accuracy].index.tolist()
        logging.info(f"Selected models with accuracy above mean: {selected_model_names}")

    else:
        logging.warning("Invalid selection method. Defaulting to diverse model selection.")
        selected_model_names = select_diverse_models(metrics, predictions, y_test, top_n=3)

    logging.info(f"Selected models for ensemble: {selected_model_names}")
    selected_models = [(name, optimized_models[name]) for name in selected_model_names]
    

    # Step 5: Ensemble voting with diverse models
    logging.info("Creating ensemble with diverse models")
    ensemble_hardVoting = VotingClassifier(estimators=selected_models, voting='hard') # Hard Voting Classifier
    ensemble_softVoting = VotingClassifier(estimators=selected_models, voting='soft') # Soft Voting Classifier

    ensemble_hardVoting.fit(X_train, y_train)
    ensemble_softVoting.fit(X_train, y_train)
    y_pred_ensemble_hardVoting = ensemble_hardVoting.predict(X_test)
    y_pred_ensemble_softVoting = ensemble_softVoting.predict(X_test)

    # Evaluate Voting Classifiers
    logging.info("Evaluating Voting Classifiers")
    hard_voting_metrics, _ = evaluate_models(models={'Ensemble_hV': ensemble_hardVoting}, X_test=X_test, y_test=y_test)
    soft_voting_metrics, _ = evaluate_models(models={'Ensemble_sV': ensemble_softVoting}, X_test=X_test, y_test=y_test)

    ensemble_hardVoting_accuracy = ensemble_hardVoting.score(X_test, y_test)
    ensemble_softVoting_accuracy = ensemble_softVoting.score(X_test, y_test)
    logging.info(f"Ensemble (with Hard voting) Accuracy: {ensemble_hardVoting_accuracy}")
    logging.info(f"Ensemble (with Soft voting) Accuracy: {ensemble_softVoting_accuracy}")

    # Save Ensemble/Voting Classifiers
    logging.info("Saving Voting Classifiers")
    save_model(ensemble_hardVoting, 'Ensemble_hV', models_dir)
    save_model(ensemble_softVoting, 'Ensemble_sV', models_dir)

     # Combine metrics from voting classifiers with other models and save
    logging.info("Saving combined metrics for all models")
    combined_metrics = pd.concat([metrics, hard_voting_metrics, soft_voting_metrics])
    metrics_dict[f"{classification_type}_{comparison}"] = combined_metrics
    print("Model Performance Evaluation Metrics:")
    print(combined_metrics.sort_values(by='Accuracy', ascending=False))

    # Combine all predictions, including those from the ensemble models
    combined_predictions = {**predictions, 'Ensemble_hV': y_pred_ensemble_hardVoting, 'Ensemble_sV': y_pred_ensemble_softVoting}
    # Save final evaluation results
    save_results(y_test, combined_predictions, os.path.join(results_dir, 'predictions.csv'))



def parse_arguments():
    parser = argparse.ArgumentParser(description="Train and evaluate models for all classification types")
    parser.add_argument('--output_dir', type=str, default='Project_Output_combined', help="Main project directory for clinical AD dataset")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = parse_arguments()

    # Dictionary to hold metrics for each classification type
    metrics_dict = {}

    # Loop over all classification types and comparisons
    for classification_type, comparisons in CLASSIFICATION_COMPARISONS.items():
        for comparison in comparisons:
            logging.info(f"###############################################################")
            logging.info(f"Processing {classification_type} classification: {comparison}")
            logging.info(f"###############################################################")

            # Create directory structure for the current classification and comparison
            data_dir, models_dir, results_dir = create_directory_structure(args.output_dir, classification_type, comparison)

            # Train and evaluate models for the current comparison
            train_and_evaluate_models(data_dir, models_dir, results_dir, classification_type, comparison, metrics_dict, selection_method='above_mean_accuracy')

    # Save all metrics to a single Excel file
    output_file = os.path.join(args.output_dir, 'classification_metrics_GT.xlsx')
    save_metrics_to_excel(metrics_dict, output_file)

if __name__ == "__main__":
    main()
