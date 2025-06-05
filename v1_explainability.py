# Copyright: Shahzad Ali
# Author: Shahzad Ali
# e-mail: shahzad.ali6@unibo.it
# Created: 2024-10-24
# Last modified: 2024-11-05

"""
This module generates various plots for model comparison as well as explainability using SHAP and LIME.
It loads trained models and test data, generates SHAP and LIME explanations,
and saves the resulting visualizations and explanations to the specified output directory.
"""

# Before using this script, run `preprocessing.py` & 'ensembleClassifier_training.py'.


import os
import argparse
import logging
import pandas as pd
from utils import load_model, load_data
import v1_2_visualization as viz
from models import select_top_models

import warnings
warnings.filterwarnings('ignore')


def get_class_mapping(comparison):
    if comparison == 'CN_vs_AD':
        return {0: 'CN', 1: 'AD'}
    elif comparison == 'CN_vs_MCI':
        return {0: 'CN', 1: 'MCI'}
    elif comparison == 'MCI_vs_AD':
        return {0: 'MCI', 1: 'AD'}
    elif comparison == 'CN_MCI_AD':
        return {0: 'CN', 1: 'MCI', 2: 'AD'}
    return {}


def generate_comparison_plots(data_dir, models_dir, results_dir, plots_dir, output_dir, classification_type, comparison):
    # Load the evaluation metrics from the Excel file
    #metrics_file = os.path.join(results_dir, 'metrics.csv')
    #eval_metrics = pd.read_csv(metrics_file, index_col=0)
    metrics_file = os.path.join(output_dir, 'classification_metrics.xlsx')  # Adjusted to access main directory
    logging.info(f"Loading evaluation metrics from {metrics_file}")

    # The sheet name corresponds to "{classification_type}_{comparison}"
    sheet_name = f"{classification_type}_{comparison}"
    eval_metrics = pd.read_excel(metrics_file, sheet_name=sheet_name)
    print("Evaluation Metrics:\n", eval_metrics)


    # Extract the model names and load the trained models
    logging.info("Loading trained models")
    model_names = eval_metrics['Model'].tolist()
    print(model_names)
    all_models = {name: load_model(name, models_dir) for name in model_names}

    # Confusion Matrices - Generate and save cm for the top models
    logging.info(f"Selecting top models for Confusion Matrix Plot")
    top_models = select_top_models(eval_metrics, all_models, top_n=2)
    logging.info(f"Top models selected for Confusion Matrix Plot: {[name for name, _ in top_models]}")

    # Load test data to generate confusion matrices
    X_test = load_data(os.path.join(data_dir, 'X_test.csv'))
    y_test = load_data(os.path.join(data_dir, 'y_test.csv')).values.ravel()

    # Generate and save confusion matrices for the top models
    logging.info("Generating confusion matrix plots for the top models")
    top_model_dict = {name: model for name, model in top_models}
    viz.plot_confusion_matrix(top_model_dict, X_test, y_test, plots_dir)
    logging.info("Model comparison and plot generation complete.")  




    logging.info("Generating model performance comparison plot")
    viz.plot_model_performance_comparison(metrics=eval_metrics, model_names=model_names, output_dir=plots_dir, filename_prefix="Comparison_ModelEvaluationMetrics")

    logging.info("Generating model accuracy plot of all models")
    viz.plot_model_accuracy(metrics=eval_metrics, model_names=model_names, output_dir=plots_dir, filename_prefix="Comparison_ModelAccuracy")

    logging.info("Generating Type I and Type II error plots for all models")
    viz.plot_type1_type2_errors(metrics=eval_metrics, model_names=model_names, output_dir=plots_dir)

    # Load the trained Decision Tree model
    dt_model_name = 'DT'
    logging.info(f"Loading the trained Decision Tree '{dt_model_name}' model")
    #dt_model = trained_models['DT']
    dt_model = load_model(model_name=dt_model_name, input_dir=models_dir)
    feature_names = pd.read_csv(os.path.join(data_dir, 'X_train.csv')).columns

    logging.info(f"Generating the decision tree plot using '{dt_model_name}'")
    viz.plot_decision_tree(model=dt_model, feature_names=feature_names, output_dir=plots_dir)

    # Load the trained Random Forest model
    rf_model_name = 'RF'
    logging.info(f"Loading the trained '{rf_model_name}' model")
    rf_model = load_model(model_name=rf_model_name, input_dir=models_dir)

    logging.info(f"Plotting feature importance for Random Forest '{rf_model_name}'")
    viz.plot_feature_importance(model=rf_model, feature_names=feature_names, top_n=15, results_dir=results_dir, output_dir=plots_dir)
 

def generate_shap_explainability(data_dir, models_dir, results_dir, plots_dir, class_mapping):
    # Load the trained Decision Tree model
    dt_model_name = 'DT'
    logging.info(f"Loading the trained Decision Tree '{dt_model_name}' model")
    dt_model = load_model(model_name=dt_model_name, input_dir=models_dir)

    # Load the training and test data
    X_train = load_data(os.path.join(data_dir, 'X_train.csv'))
    X_test = load_data(os.path.join(data_dir, 'X_test.csv'))
    y_test = load_data(os.path.join(data_dir, 'y_test.csv')).values.ravel()

    logging.info("############ Explainability using SHAP ############")

    logging.info(f"Plotting SHAP summary for each class using {dt_model_name}")
    viz.plot_shap_summary_by_class(model=dt_model, X_test=X_test, class_mapping=class_mapping, output_dir=plots_dir)

    logging.info(f"Plotting aggregated SHAP summary using {dt_model_name}")
    viz.plot_shap_aggregated_summary(model=dt_model, X_test=X_test, output_dir=plots_dir)

    #logging.info(f"Plotting SHAP dependence for the feature 'CDRSB_bl' using {dt_model_name}")
    #viz.plot_shap_dependence(model=dt_model, X_test=X_test, feature='CDRSB_bl', output_dir=plots_dir)

    #logging.info("Plotting SHAP dependence and feature importance for all features")
    viz.plot_shap_dependence_and_feature_importance(model=dt_model, X_test=X_test, class_mapping=class_mapping, output_dir=plots_dir, results_dir=results_dir)

    logging.info(f"Plotting SHAP force plot for a specific sample using {dt_model_name}")
    viz.plot_force_for_sample(model=dt_model, X_test=X_test, sample_index=0, output_dir=plots_dir)

    logging.info(f"Plotting SHAP decision plot for a specific sample and class using {dt_model_name}")
    viz.plot_decision_for_sample_and_class(model=dt_model, X_test=X_test, sample_index=0, class_index=0, class_mapping=class_mapping, output_dir=plots_dir)

    logging.info(f"Plotting SHAP decision plots for all samples for each class using {dt_model_name}")
    viz.plot_decision_for_all_samples(model=dt_model, X_test=X_test, class_mapping=class_mapping, output_dir=plots_dir)

    logging.info("############ Explainability using LIME ############")
    logging.info(f"Generating LIME explanation for a specific sample using {dt_model_name}")
    viz.plot_lime_explanation(model=dt_model, X_train=X_train, X_test=X_test, class_mapping=class_mapping, index=0, num_features=15, output_dir=plots_dir)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate visualizations and explainability for models")
    parser.add_argument('--output_dir', type=str, default='Project_Output_GT_test', help="Main project directory for clinical AD dataset")
    return parser.parse_args()

def main():
    # Configure logging to display INFO level messages
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = parse_arguments()

    # Define all classification types and comparisons (same as in preprocessing.py and training.py)
    CLASSIFICATION_COMPARISONS = {
        'binary': ['CN_vs_AD', 'CN_vs_MCI', 'MCI_vs_AD'],
        'three_level': ['CN_MCI_AD']
    }
    # Loop over all classification types and comparisons
    for classification_type, comparisons in CLASSIFICATION_COMPARISONS.items():
        for comparison in comparisons:
            logging.info(f"###############################################################")
            logging.info(f"Processing {classification_type} classification: {comparison}")
            logging.info(f"###############################################################")

            # Set up directories based on the comparison
            if comparison != 'CN_MCI_AD':
                folder_name = comparison.replace('vs_', '')
            else:
                folder_name = 'CN_MCI_AD'

            data_dir = os.path.join(args.output_dir, folder_name, 'data')
            models_dir = os.path.join(args.output_dir, folder_name, 'models')
            results_dir = os.path.join(args.output_dir, folder_name, 'results')
            plots_dir = os.path.join(args.output_dir, folder_name, 'plots')

            # Ensure directories exist
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(models_dir, exist_ok=True)
            os.makedirs(results_dir, exist_ok=True)
            os.makedirs(plots_dir, exist_ok=True)

            # Get the correct class mapping based on the comparison
            class_mapping = get_class_mapping(comparison)

            # Generate model comparison plots and SHAP explainability
            generate_comparison_plots(data_dir, models_dir, results_dir, plots_dir, args.output_dir, classification_type, comparison)
            generate_shap_explainability(data_dir, models_dir, results_dir, plots_dir, class_mapping)


if __name__ == "__main__":
    main()
