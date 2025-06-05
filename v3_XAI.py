# Copyright: Shahzad Ali
# Author: Shahzad Ali
# e-mail: shahzad.ali6@unibo.it
# Created: 2025-03-21
# Last modified: 2025-04-14

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from sklearn.ensemble import AdaBoostRegressor

import shap
from lime import lime_tabular


###########################################
#      SHAP Explainability Plot Functions
###########################################

def plot_shap_summary(model, X_test, output_dir):
    """ Generates a SHAP summary plot for a regression model. """
    if not hasattr(model, 'predict'):
        logging.warning("The provided model does not have a predict method.")
        return

    # Use TreeExplainer for tree-based models
    if hasattr(model, 'tree_') or isinstance(model, shap.explainers._tree.TreeExplainer):
        explainer = shap.TreeExplainer(model)  # Instantiate without 'check_additivity'
        shap_values = explainer.shap_values(X_test, check_additivity=False)  # Pass 'check_additivity' here
    elif isinstance(model, AdaBoostRegressor):
        # Use KernelExplainer for AdaBoost
        explainer = shap.KernelExplainer(model.predict, X_test)
        shap_values = explainer.shap_values(X_test)
    else:
        # General SHAP Explainer for non-tree models
        explainer = shap.Explainer(model, X_test)
        shap_values = explainer(X_test)

    # Generate SHAP summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(f'{output_dir}/Shap_Summary_{model.__class__.__name__}.png', format='png', bbox_inches='tight', dpi=500)
    plt.close()
    logging.info(f"SHAP summary plot saved to {output_dir}.")



def plot_shap_dependence(model, X_test, feature, output_dir):
    """ Generates a SHAP dependence plot for a specific feature in a regression model. """
    if not hasattr(model, 'predict'):
        logging.warning("Model does not have a predict method.")
        return

    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)

    plt.figure()
    shap.dependence_plot(feature, shap_values.values, X_test, show=False)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/SHAP_Dependence_{feature}_{model.__class__.__name__}.png', format='png', bbox_inches='tight', dpi=500)
    plt.close()
    logging.info(f"SHAP dependence plot saved to {output_dir}.")

def plot_shap_feature_importance(model, X_test, feature_names, output_dir, results_dir):
    """ Generates a SHAP-based feature importance plot for a regression model. """

    if not hasattr(model, 'predict'):
        logging.warning("The provided model does not have a predict method.")
        return

    # Use KernelExplainer for AdaBoost and other models that are not directly callable
    if isinstance(model, AdaBoostRegressor):
        explainer = shap.KernelExplainer(model.predict, X_test)
        shap_values = explainer.shap_values(X_test)
    else:
        # General SHAP Explainer for other models
        explainer = shap.Explainer(model, X_test)
        shap_values = explainer(X_test)

    # Compute mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Create a DataFrame to store the feature importances
    shap_importances = pd.DataFrame({
        "Feature": feature_names,
        "Importance": mean_abs_shap
    }).sort_values(by="Importance", ascending=False)

    # Save SHAP feature importances to a CSV file
    csv_path = os.path.join(results_dir, 'Shap_Feature_Importance.csv')
    shap_importances.to_csv(csv_path, index=False)
    logging.info(f"SHAP Feature Importance saved to {csv_path}")
    
    # Plot SHAP feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x="Feature", y="Importance", data=shap_importances.head(15), palette="coolwarm")
    plt.xlabel("Feature")
    plt.ylabel("Mean SHAP Value")
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.title("SHAP Feature Importances")
    plt.tight_layout()

    plt.savefig(f'{output_dir}/Shap_Feature_Importance_{model.__class__.__name__}.png', format='png', bbox_inches='tight', dpi=500)
    plt.close()
    logging.info(f"SHAP feature importance plot saved to {output_dir}.")


###########################################
#      LIME Explainability Plot Functions
###########################################
def plot_lime_explanation(model, X_train, X_test, sample_index, num_features, output_dir):
    """ Generate and save a LIME explanation for a specific sample in a regression model. """
    if not hasattr(model, 'predict'):
        logging.warning("Model does not have a predict method.")
        return

    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        mode="regression",
        discretize_continuous=True
    )

    explanation = explainer.explain_instance(
        data_row=X_test.iloc[sample_index].values,
        predict_fn=model.predict,
        num_features=num_features
    )

    os.makedirs(output_dir, exist_ok=True)
    explanation.save_to_file(f"{output_dir}/LIME_Explanation_Sample_{sample_index}.html")
    explanation.as_pyplot_figure()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/LIME_Explanation_Sample_{sample_index}.png', format='png', bbox_inches='tight', dpi=500)
    plt.close()
    logging.info(f"LIME explanation for sample {sample_index} saved to {output_dir}.")

    # Optionally, adjust plot size and save a high-quality plot as PNG
    fig = explanation.as_pyplot_figure()
    fig.set_size_inches(12, 10)  # Adjust figure size for better readability
    plt.subplots_adjust(left=0.3)  # Increase left margin
    fig.savefig(f'{output_dir}/lime_explanation_{model.__class__.__name__}.png', dpi=500)
    plt.close(fig)  # Close the figure to avoid display issues in Jupyter notebooks
