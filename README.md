# A Systematic Study on the Integration of Advanced MRI Connectivity Metrics for Alzheimer’s Disease Diagnosis and Staging and Longitudinal Cognitive Decline Prediction

## Project Overview
This project implements a machine learning pipeline for the Alzheimer's Disease Stage classification (DSC) and Longitudinal Cognitive Decline Prediction (LCDP) in terms of delta MMSE. The pipeline supports both binary classification (CN-ADD, CN-MCI, MCI-ADD) and three-level classification (CN-MCI-ADD) using various machine learning models. It includes preprocessing of multimodal data (demographic, Morphometric (MO), Miscrostructial (MO), and Graph Theory (GT) features), model training, optimization, Ensemble Learning, eXplainability (XAI).




## Project Structure
- **preprocessing.py**: Handles the preprocessing of clinical data, including feature scaling and train-test splitting.
- **model_optimization.py**: Selects top-performing models and optimizes their hyperparameters via five-fold CV using GridSearchCV.
- **model_training.py**: Trains various machine learning models and evaluates their performance on the test set.
- **voting.py**: Implements ensemble classifier using the top-performing models from both the optimized models.
- **final_comparison.py**: Compares the performance of all models (initial, optimized, and voting classifiers) and generates visualizations.
- **main.py**: Orchestrates the entire pipeline, running all steps in sequence.

## Directory Structure
Upon execution, the project will create and organize the following directory structure:
- **data/**: Contains preprocessed training and test datasets.
- **models/**: Stores trained and optimized machine learning models.
- **results/**: Contains evaluation metrics and predictions.
- **plots/**: Stores generated comparison plots and confusion matrices.

ProjectOutput_<Feature_Combination>/
├── CN_vs_AD/               # Classification comparison folder (e.g., CN vs AD)
│   ├── data/               # Folder to store training and test datasets (X_train, X_test, y_train, y_test)
│   ├── models/             # Folder to save trained models
│   ├── plots/              # Folder to save generated plots
│   └── results/            # Folder to save final results
├── CN_vs_MCI/              # Another comparison (e.g., CN vs MCI)
│   ├── data/
│   ├── models/
│   ├── plots/
│   └── results/
└── CN_MCI_AD/              # Three-level classification folder
    ├── data/
    ├── models/
    ├── plots/
    └── results/


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
##### For Clinical Data:
```
python preprocessing.py 
```
##### For Multimodal Data (Clinical + Graph Theory Metrics + TCK Metrics)
```
python preprocessing_multimod.py 
```
#### Step 2: Model Training
```
python model_training.py 
```
#### Step 3: Final Model Comparison
```
python final_comparison.py
```
#### Step 4: Model Optimization
```
python model_optimization.py 
```
#### Step 5: Voting Classifiers
```
python voting.py 
```
#### Step Final Comparison:
```
python final_comparison.py 
```

## Option 2: Full Pipeline Execution
Alternatively, you can execute the entire pipeline from preprocessing to final comparison in one go using the main.py script:
```
python main.py
```


## Contributing
Feel free to submit issues or pull requests if you have any improvements or bug fixes.
