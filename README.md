# A Systematic Study on the Integration of Advanced MRI Connectivity Metrics for Alzheimer’s Disease Diagnosis and Staging and Longitudinal Cognitive Decline Prediction

## Project Overview
This project implements a machine learning pipeline for the Alzheimer's Disease Stage classification (DSC) and Longitudinal Cognitive Decline Prediction (LCDP) in terms of delta MMSE. The pipeline supports both binary classification (CN-ADD, CN-MCI, MCI-ADD) and multi-level classification (CN-MCI-ADD) using various machine learning models. It includes preprocessing and preparing data (demographic, Morphometric (MO), Miscrostructial (MO), and Graph Theory (GT) features), model training and optimization, ensemble Learning, eXplainability (XAI).




## Project Structure
- **preprocessing.py**: handles the preprocessing of clinical data, including feature scaling and train-test splitting w.r.t. each task.
- **model_optimization.py**: performs model's hyperparameter tunning via five-fold CV using GridSearchCV and selects top-performing optmized mdoels. Consequently, train models and evaluates their performance on the test set.
- **ensemble.py**: Implements ensemble classifier from the best performing base models.
- **explainabiility.py**: Compares the performance of all models (initial, optimized, and voting classifiers) and generates visualizations including, confusion matrices, feature importance charts, XAI based plots (using SHAP and LIME).
- **main.py**: Orchestrates the entire pipeline, running all steps in sequence.

## Directory Structure
Upon execution, the project will create and organize the following directory structure:
- **data/**: Contains preprocessed training and test datasets.
- **models/**: Stores trained and optimized machine learning models.
- **results/**: Contains evaluation metrics and predictions.
- **plots/**: Stores generated comparison plots and confusion matrices.


## How to Run the Project
### Clone the repository
```
git clone <https://github.com/shahzadali21/MRI_Connectivity_Metrics_for_AD_Diagnosis_and_Staging.git>
cd <repository-directory>
```
### Install dependencies
- Python 3.7+
- Ensure you have all the required Python packages installed. You can install them using the following command:
```
pip install -r requirements.txt
```

## Running the Project
### Option 1: Running the Project Step-by-Step
#### Step 1: Data Preprocessing
```
python preprocessing.py
```
#### Step 2: Model Training
```
python model_optimizing.py 
```
#### Step 3: Final Model Comparison
```
python ensemble.py
```
#### Step 4: Model Optimization
```
python explainability.py 
```

## Option 2: Full Pipeline Execution
Alternatively, you can execute the entire pipeline from preprocessing to final comparison in one go using the main.py script:
```
python main.py
```


## Contributing
Feel free to submit issues or pull requests if you have any improvements or bug fixes.
