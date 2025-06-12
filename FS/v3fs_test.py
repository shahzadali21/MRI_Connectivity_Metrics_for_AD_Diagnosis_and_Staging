

import os
import logging
import glob
import pandas as pd

from v3fs_feature_selection import feature_selection
from v3fs_model_optimizing import train_and_evaluate_models


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Feature combination to use
feature_combination = "Demographic_Microstructural_Morphometric_GT"

# Directories and Configuration
data_dir = "V3fs_ProjectOutput/Demographic_Microstructural_Morphometric_GT/without_nan/data"
output_dir = "V3fs_ProjectOutput"
summary_file = os.path.join(output_dir, "metrics_summary.xlsx")


# Load base data
X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()

# Feature selection methods to test
feature_selection_methods = ["mutual_info", "anova", "rfe_elastic_net", "random_forest", "pca"]

# Perform feature selection and train models iteratively
for method in feature_selection_methods:
    logging.info(f"Starting feature selection method: {method}")
    method_dir = os.path.join(output_dir, method)
    os.makedirs(method_dir, exist_ok=True)
    
    for k in range(20, X_train.shape[1] + 1, 20):  # Increment by 20 features
        logging.info(f"Feature selection method: {method}, Top {k} features")
        
        # Feature selection
        X_train_selected, selected_features = feature_selection(X_train, y_train, method=method, k=k)
        X_test_selected = X_test.iloc[:, selected_features]

        # Create directories for this iteration
        iteration_dir = os.path.join(method_dir, f"top_{k}")
        os.makedirs(iteration_dir, exist_ok=True)
        models_dir = os.path.join(iteration_dir, "models")
        results_dir = os.path.join(iteration_dir, "results")

        # Save reduced datasets
        data_dir_iteration = os.path.join(iteration_dir, "data")
        os.makedirs(data_dir_iteration, exist_ok=True)

        pd.DataFrame(X_train_selected, columns=X_train.columns[selected_features]).to_csv(
            os.path.join(data_dir_iteration, "X_train.csv"), index=False)
        pd.DataFrame(X_test_selected, columns=X_test.columns[selected_features]).to_csv(
            os.path.join(data_dir_iteration, "X_test.csv"), index=False)

        pd.DataFrame(y_train).to_csv(os.path.join(data_dir_iteration, "y_train.csv"), index=False)
        pd.DataFrame(y_test).to_csv(os.path.join(data_dir_iteration, "y_test.csv"), index=False)

        # Train and evaluate models
        metrics_dict = {}
        train_and_evaluate_models(
            data_dir=data_dir_iteration,
            models_dir=models_dir,
            results_dir=results_dir,
            subset_name=f"top_{k}",
            metrics_dict=metrics_dict,
            feature_selection_method=method,
            k=k
        )

        # Save individual metrics to a CSV
        metrics_file = os.path.join(results_dir, f"metrics_top_{k}.csv")
        #metrics_df = pd.DataFrame.from_dict(metrics_dict[f"top_{k}"], orient="index")
        metrics_df = metrics_dict[f"top_{k}"]
        metrics_df.to_csv(metrics_file, index=True)
        logging.info(f"Saved metrics for {method}, Top {k} features to {metrics_file}")

# Generate a summary of all metrics
logging.info("Generating summary of all metrics...")
summary = []

for method in feature_selection_methods:
    method_dir = os.path.join(output_dir, method)
    for metrics_file in glob.glob(f"{method_dir}/top_*/results/metrics_top_*.csv"):
        df = pd.read_csv(metrics_file)
        # Extract method and top_k information from the file path
        k = int(metrics_file.split("top_")[1].split("/")[0])
        df["Method"] = method
        df["Top K"] = k
        summary.append(df)

# Combine all metrics into a single DataFrame
summary_df = pd.concat(summary, ignore_index=True)

# Save the summary to an Excel file
summary_df.to_excel(summary_file, index=False)
logging.info(f"Metrics summary saved to {summary_file}")
print(f"Metrics summary saved to {summary_file}")