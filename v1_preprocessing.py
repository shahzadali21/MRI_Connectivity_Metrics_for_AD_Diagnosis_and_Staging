# Copyright: Shahzad Ali
# Author: Shahzad Ali
# e-mail: shahzad.ali6@unibo.it
# Created: 2024-10-24
# Last modified: 2024-10-24
# Version: 1.0
"""
This script handles the preprocessing of clinical data.
It reads the data, performs basic EDA, extracts features and targets,
scales numeric features for training and test sets separately for all classification types,
and saves the final processed data to the specified output directory structure.
"""

# Before using this script, first prepare the dataset using `1_AD_EDA.ipynb` available in the notebook.

import os
import logging
import argparse
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from utils import save_data



def preprocess_data(dataframe, classification_type, comparison, scaler_type, project_output_dir, data_dir, test_size=0.20, seed=42, preprocessed=False, use_smote=False):
    if not preprocessed:
        logging.info(f"################################################################################")
        logging.info(f"########################### Starting preprocessing... #########################")
        logging.info(f"################################################################################")
        logging.info(f"############################### OVERVIEW OF SETUP ##############################")
        logging.info(f"\nClassification Type: {classification_type}, \nComparison: {comparison}, \nScaler Type: {scaler_type}, \nOutput Directory: {project_output_dir}, \nData Directory: {data_dir}, \nTest Size: {test_size}, \nSeed: {seed}")
        logging.info(f"Shape of dataframe: {dataframe.shape}, \nValue counts of DX: {dataframe['DX'].value_counts()}, \nValue counts of Research Group: {dataframe['RG'].value_counts()}")
        logging.info(f"SETUP Information: \nDataframe: {dataframe.head()}")
        save_data(dataframe, os.path.join(project_output_dir, 'ADNI_CTCKGT_mmse.csv'))
        logging.info(f"################################################################################")
        logging.info(f"############################# PRE-PROECESSING DATA #############################")
        logging.info(f"################################################################################")

        #if comparison == 'comparison':

        # Keep original df for backup
        # dataframe_all = dataframe.copy()

        logging.info(f"\n\n######################## REMOVING UNNECESSARY COLUMNS #####################")
        unnecessary_cols = ['RID', 'COLPROT', 'ORIGPROT', 'SITE', 'Final_Status', 'DX_bl', 'Transition_DXbl_DX', 'Transition_DXbl_Group', 'EXAMDATE_bl', 'EXAMDATE', 'AGE_bl']
        dataframe.drop(columns=unnecessary_cols, inplace=True)
        logging.info(f"Removed unnecessary columns: {unnecessary_cols}")
        logging.info(f"Shape of dataframe after removing unnecessary columns: {dataframe.shape}")
        logging.info(f"Columns in dataframe after removing unnecessary columns: {list(dataframe.columns)}")


        logging.info(f"\n\n############################## LABEL ENCODING ############################")
        # Optional - Convert columns to 'category' type (considering 'APOE4' as cateogory)
        columns_to_convert = ['APOE4']     # define list of columns to consider as category
        dataframe[columns_to_convert] = dataframe[columns_to_convert].astype('category')
        #categorical_cols = dataframe.select_dtypes(include=['object', 'category']).columns
        categorical_cols = dataframe.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
        numerical_cols = dataframe.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Inspect categories for each categorical column
        for column in categorical_cols:
            if column == 'Subject ID':
                continue
            unique_values = dataframe[column].unique()
            print(f"Unique values in {column}: {unique_values}")
        
        logging.info(f"List of Categorical Columns: {categorical_cols}")
        logging.info(f"List of Numerical Columns: {numerical_cols}")

        #Encoding Workflow:
        #- Drop `Subject ID`: Subject ID should be dropped after merging datasets.
        #- `One-Hot Encoding for Nominal Variables`: Use one-hot encoding for **`Sex`** and **`PTMARRY`**.
        #- `Label Encoding for Ordinal Variables`: Use label encoding for the ordinal variable **`DX` / `RG`** (stages of AD).

        le = LabelEncoder()
        #df['RG_le'] = le.fit_transform(df['RG'])  # defines automatically (CN -> 1, MCI -> 2, AD -> 0)
        #df = df.drop(['RG_le', 'DX_le'], axis=1)

        # Manually define the mapping for RG and DX based on desired order
        sex_map = {'F': 0, 'M': 1}
        marry_map = {'Never married': 0, 'Married': 1, 'Widowed': 2, 'Divorced': 3, 'Unknown': 4}
        rg_dx_map = {'CN': 0, 'MCI': 1, 'AD': 2}

        # Apply the mapping
        #dataframe['DX_le'] = dataframe['DX'].map(rg_dx_map)
        # Perform label encoding and insert new columns right after the original columns in one go
        dataframe.insert(dataframe.columns.get_loc('Sex') + 1, 'Sex_le', dataframe['Sex'].map(sex_map))
        dataframe.insert(dataframe.columns.get_loc('PTMARRY') + 1, 'PTMARRY_le', dataframe['PTMARRY'].map(marry_map))
        dataframe.insert(dataframe.columns.get_loc('RG') + 1, 'RG_le', dataframe['RG'].map(rg_dx_map))
        dataframe.insert(dataframe.columns.get_loc('DX') + 1, 'DX_le', dataframe['DX'].map(rg_dx_map))

        """
        # One-Hot Encoding for 'Sex' and 'PTMARRY': Since these are nominal (non-ordinal) variables with no inherent order.
        # Generate dummy variables without dropping the original columns
        # Generate dummy variables and insert them next to the original columns
        for col in ['Sex', 'PTMARRY']:
            dummies = pd.get_dummies(dataframe[col], drop_first=True)  # Keep all original levels
            dummies.columns = [f"{col}_{val}" for val in dummies.columns]  # Rename columns
            
            for dummy_col in dummies.columns:
                # Insert each dummy column right after the original column
                dataframe.insert(dataframe.columns.get_loc(col) + 2, dummy_col, dummies[dummy_col])
        """
        logging.info(f"\nShape of dataframe after Label Encoding: {dataframe.shape}")
        logging.info(f"Columns in dataframe after Label Encoding: {list(dataframe.columns)}")


        # Save the Label Encoded data
        save_data(dataframe, os.path.join(project_output_dir, 'ADNI_CTCKGT_mmse_le.csv'))
        logging.info(f"Data saved after Label Encoding to {project_output_dir}")


        logging.info(f"\n\n################## SPLITTING DATA INTO CATEGORIES ##################")
        ## CATEGORIZATION - (Demographic | Clinical | GT | MS Features)
        demographic_cols = ['Subject ID', 'RG', 'RG_le', 'DX', 'DX_le', 'deltaMMSE/Years', 'Age', 'PTEDUCAT', 'Sex', 'Sex_le','PTMARRY',  'PTMARRY_le']  # 'Sex_M', 'PTMARRY_Widowed', 'PTMARRY_Never married', 'PTMARRY_Married', 'PTMARRY_Unknown' (only if they are one-hot encoded)
        clinical_cols = ['Subject ID', 'DX_le', 'RG_le', 'MMSE_bl', 'MMSE', 'MOCA_bl', 'MOCA', 'APOE4', 'CDRSB_bl', 'CDRSB', 'ADAS11_bl', 'ADAS11', 'ADAS13_bl', 'ADAS13', 'ADASQ4_bl', 'ADASQ4']
        graphtheory_cols = ['Subject ID', 'DX_le', 'RG_le', 'density', 'modularity', 'assortativity', 'transitivity', 'global_efficiency', 'characteristic_path_length', 'diameter', 'degree_distribution_entropy', 'resilience', 'spectral_radius', 'avg_clustering_coefficient', 'small_worldness', 'avg_degree', 'avg_betweenness_centrality', 'avg_edge_betweenness_centrality', 'avg_eigenvector_centrality', 'avg_closeness_centrality', 'avg_node_strength', 'avg_pagerank', 'degree_centrality_node_0', 'degree_centrality_node_1', 'degree_centrality_node_2', 'degree_centrality_node_3', 'degree_centrality_node_4', 'degree_centrality_node_5', 'degree_centrality_node_6', 'degree_centrality_node_7', 'degree_centrality_node_8', 'degree_centrality_node_9', 'degree_centrality_node_10', 'degree_centrality_node_11', 'degree_centrality_node_12', 'degree_centrality_node_13', 'degree_centrality_node_14', 'degree_centrality_node_15', 'degree_centrality_node_16', 'degree_centrality_node_17', 'degree_centrality_node_18', 'degree_centrality_node_19', 'degree_centrality_node_20', 'degree_centrality_node_21', 'degree_centrality_node_22', 'degree_centrality_node_23', 'degree_centrality_node_24', 'degree_centrality_node_25', 'degree_centrality_node_26', 'degree_centrality_node_27', 'degree_centrality_node_28', 'degree_centrality_node_29', 'degree_centrality_node_30', 'degree_centrality_node_31', 'degree_centrality_node_32', 'degree_centrality_node_33', 'degree_centrality_node_34', 'degree_centrality_node_35', 'degree_centrality_node_36', 'degree_centrality_node_37', 'degree_centrality_node_38', 'degree_centrality_node_39', 'degree_centrality_node_40', 'degree_centrality_node_41', 'degree_centrality_node_42', 'degree_centrality_node_43', 'degree_centrality_node_44', 'degree_centrality_node_45', 'degree_centrality_node_46', 'degree_centrality_node_47', 'degree_centrality_node_48', 'degree_centrality_node_49', 'degree_centrality_node_50', 'degree_centrality_node_51', 'degree_centrality_node_52', 'degree_centrality_node_53', 'degree_centrality_node_54', 'degree_centrality_node_55', 'degree_centrality_node_56', 'degree_centrality_node_57', 'degree_centrality_node_58', 'degree_centrality_node_59', 'degree_centrality_node_60', 'degree_centrality_node_61', 'degree_centrality_node_62', 'degree_centrality_node_63', 'degree_centrality_node_64', 'degree_centrality_node_65', 'degree_centrality_node_66', 'degree_centrality_node_67', 'degree_centrality_node_68', 'degree_centrality_node_69', 'degree_centrality_node_70', 'degree_centrality_node_71', 'degree_centrality_node_72', 'degree_centrality_node_73', 'degree_centrality_node_74', 'degree_centrality_node_75', 'degree_centrality_node_76', 'degree_centrality_node_77', 'degree_centrality_node_78', 'degree_centrality_node_79', 'degree_centrality_node_80', 'degree_centrality_node_81', 'clustering_coefficient_node_0', 'clustering_coefficient_node_1', 'clustering_coefficient_node_2', 'clustering_coefficient_node_3', 'clustering_coefficient_node_4', 'clustering_coefficient_node_5', 'clustering_coefficient_node_6', 'clustering_coefficient_node_7', 'clustering_coefficient_node_8', 'clustering_coefficient_node_9', 'clustering_coefficient_node_10', 'clustering_coefficient_node_11', 'clustering_coefficient_node_12', 'clustering_coefficient_node_13', 'clustering_coefficient_node_14', 'clustering_coefficient_node_15', 'clustering_coefficient_node_16', 'clustering_coefficient_node_17', 'clustering_coefficient_node_18', 'clustering_coefficient_node_19', 'clustering_coefficient_node_20', 'clustering_coefficient_node_21', 'clustering_coefficient_node_22', 'clustering_coefficient_node_23', 'clustering_coefficient_node_24', 'clustering_coefficient_node_25', 'clustering_coefficient_node_26', 'clustering_coefficient_node_27', 'clustering_coefficient_node_28', 'clustering_coefficient_node_29', 'clustering_coefficient_node_30', 'clustering_coefficient_node_31', 'clustering_coefficient_node_32', 'clustering_coefficient_node_33', 'clustering_coefficient_node_34', 'clustering_coefficient_node_35', 'clustering_coefficient_node_36', 'clustering_coefficient_node_37', 'clustering_coefficient_node_38', 'clustering_coefficient_node_39', 'clustering_coefficient_node_40', 'clustering_coefficient_node_41', 'clustering_coefficient_node_42', 'clustering_coefficient_node_43', 'clustering_coefficient_node_44', 'clustering_coefficient_node_45', 'clustering_coefficient_node_46', 'clustering_coefficient_node_47', 'clustering_coefficient_node_48', 'clustering_coefficient_node_49', 'clustering_coefficient_node_50', 'clustering_coefficient_node_51', 'clustering_coefficient_node_52', 'clustering_coefficient_node_53', 'clustering_coefficient_node_54', 'clustering_coefficient_node_55', 'clustering_coefficient_node_56', 'clustering_coefficient_node_57', 'clustering_coefficient_node_58', 'clustering_coefficient_node_59', 'clustering_coefficient_node_60', 'clustering_coefficient_node_61', 'clustering_coefficient_node_62', 'clustering_coefficient_node_63', 'clustering_coefficient_node_64', 'clustering_coefficient_node_65', 'clustering_coefficient_node_66', 'clustering_coefficient_node_67', 'clustering_coefficient_node_68', 'clustering_coefficient_node_69', 'clustering_coefficient_node_70', 'clustering_coefficient_node_71', 'clustering_coefficient_node_72', 'clustering_coefficient_node_73', 'clustering_coefficient_node_74', 'clustering_coefficient_node_75', 'clustering_coefficient_node_76', 'clustering_coefficient_node_77', 'clustering_coefficient_node_78', 'clustering_coefficient_node_79', 'clustering_coefficient_node_80', 'clustering_coefficient_node_81', 'betweenness_centrality_node_0', 'betweenness_centrality_node_1', 'betweenness_centrality_node_2', 'betweenness_centrality_node_3', 'betweenness_centrality_node_4', 'betweenness_centrality_node_5', 'betweenness_centrality_node_6', 'betweenness_centrality_node_7', 'betweenness_centrality_node_8', 'betweenness_centrality_node_9', 'betweenness_centrality_node_10', 'betweenness_centrality_node_11', 'betweenness_centrality_node_12', 'betweenness_centrality_node_13', 'betweenness_centrality_node_14', 'betweenness_centrality_node_15', 'betweenness_centrality_node_16', 'betweenness_centrality_node_17', 'betweenness_centrality_node_18', 'betweenness_centrality_node_19', 'betweenness_centrality_node_20', 'betweenness_centrality_node_21', 'betweenness_centrality_node_22', 'betweenness_centrality_node_23', 'betweenness_centrality_node_24', 'betweenness_centrality_node_25', 'betweenness_centrality_node_26', 'betweenness_centrality_node_27', 'betweenness_centrality_node_28', 'betweenness_centrality_node_29', 'betweenness_centrality_node_30', 'betweenness_centrality_node_31', 'betweenness_centrality_node_32', 'betweenness_centrality_node_33', 'betweenness_centrality_node_34', 'betweenness_centrality_node_35', 'betweenness_centrality_node_36', 'betweenness_centrality_node_37', 'betweenness_centrality_node_38', 'betweenness_centrality_node_39', 'betweenness_centrality_node_40', 'betweenness_centrality_node_41', 'betweenness_centrality_node_42', 'betweenness_centrality_node_43', 'betweenness_centrality_node_44', 'betweenness_centrality_node_45', 'betweenness_centrality_node_46', 'betweenness_centrality_node_47', 'betweenness_centrality_node_48', 'betweenness_centrality_node_49', 'betweenness_centrality_node_50', 'betweenness_centrality_node_51', 'betweenness_centrality_node_52', 'betweenness_centrality_node_53', 'betweenness_centrality_node_54', 'betweenness_centrality_node_55', 'betweenness_centrality_node_56', 'betweenness_centrality_node_57', 'betweenness_centrality_node_58', 'betweenness_centrality_node_59', 'betweenness_centrality_node_60', 'betweenness_centrality_node_61', 'betweenness_centrality_node_62', 'betweenness_centrality_node_63', 'betweenness_centrality_node_64', 'betweenness_centrality_node_65', 'betweenness_centrality_node_66', 'betweenness_centrality_node_67', 'betweenness_centrality_node_68', 'betweenness_centrality_node_69', 'betweenness_centrality_node_70', 'betweenness_centrality_node_71', 'betweenness_centrality_node_72', 'betweenness_centrality_node_73', 'betweenness_centrality_node_74', 'betweenness_centrality_node_75', 'betweenness_centrality_node_76', 'betweenness_centrality_node_77', 'betweenness_centrality_node_78', 'betweenness_centrality_node_79', 'betweenness_centrality_node_80', 'betweenness_centrality_node_81', 'eigenvector_centrality_node_0', 'eigenvector_centrality_node_1', 'eigenvector_centrality_node_2', 'eigenvector_centrality_node_3', 'eigenvector_centrality_node_4', 'eigenvector_centrality_node_5', 'eigenvector_centrality_node_6', 'eigenvector_centrality_node_7', 'eigenvector_centrality_node_8', 'eigenvector_centrality_node_9', 'eigenvector_centrality_node_10', 'eigenvector_centrality_node_11', 'eigenvector_centrality_node_12', 'eigenvector_centrality_node_13', 'eigenvector_centrality_node_14', 'eigenvector_centrality_node_15', 'eigenvector_centrality_node_16', 'eigenvector_centrality_node_17', 'eigenvector_centrality_node_18', 'eigenvector_centrality_node_19', 'eigenvector_centrality_node_20', 'eigenvector_centrality_node_21', 'eigenvector_centrality_node_22', 'eigenvector_centrality_node_23', 'eigenvector_centrality_node_24', 'eigenvector_centrality_node_25', 'eigenvector_centrality_node_26', 'eigenvector_centrality_node_27', 'eigenvector_centrality_node_28', 'eigenvector_centrality_node_29', 'eigenvector_centrality_node_30', 'eigenvector_centrality_node_31', 'eigenvector_centrality_node_32', 'eigenvector_centrality_node_33', 'eigenvector_centrality_node_34', 'eigenvector_centrality_node_35', 'eigenvector_centrality_node_36', 'eigenvector_centrality_node_37', 'eigenvector_centrality_node_38', 'eigenvector_centrality_node_39', 'eigenvector_centrality_node_40', 'eigenvector_centrality_node_41', 'eigenvector_centrality_node_42', 'eigenvector_centrality_node_43', 'eigenvector_centrality_node_44', 'eigenvector_centrality_node_45', 'eigenvector_centrality_node_46', 'eigenvector_centrality_node_47', 'eigenvector_centrality_node_48', 'eigenvector_centrality_node_49', 'eigenvector_centrality_node_50', 'eigenvector_centrality_node_51', 'eigenvector_centrality_node_52', 'eigenvector_centrality_node_53', 'eigenvector_centrality_node_54', 'eigenvector_centrality_node_55', 'eigenvector_centrality_node_56', 'eigenvector_centrality_node_57', 'eigenvector_centrality_node_58', 'eigenvector_centrality_node_59', 'eigenvector_centrality_node_60', 'eigenvector_centrality_node_61', 'eigenvector_centrality_node_62', 'eigenvector_centrality_node_63', 'eigenvector_centrality_node_64', 'eigenvector_centrality_node_65', 'eigenvector_centrality_node_66', 'eigenvector_centrality_node_67', 'eigenvector_centrality_node_68', 'eigenvector_centrality_node_69', 'eigenvector_centrality_node_70', 'eigenvector_centrality_node_71', 'eigenvector_centrality_node_72', 'eigenvector_centrality_node_73', 'eigenvector_centrality_node_74', 'eigenvector_centrality_node_75', 'eigenvector_centrality_node_76', 'eigenvector_centrality_node_77', 'eigenvector_centrality_node_78', 'eigenvector_centrality_node_79', 'eigenvector_centrality_node_80', 'eigenvector_centrality_node_81', 'closeness_centrality_node_0', 'closeness_centrality_node_1', 'closeness_centrality_node_2', 'closeness_centrality_node_3', 'closeness_centrality_node_4', 'closeness_centrality_node_5', 'closeness_centrality_node_6', 'closeness_centrality_node_7', 'closeness_centrality_node_8', 'closeness_centrality_node_9', 'closeness_centrality_node_10', 'closeness_centrality_node_11', 'closeness_centrality_node_12', 'closeness_centrality_node_13', 'closeness_centrality_node_14', 'closeness_centrality_node_15', 'closeness_centrality_node_16', 'closeness_centrality_node_17', 'closeness_centrality_node_18', 'closeness_centrality_node_19', 'closeness_centrality_node_20', 'closeness_centrality_node_21', 'closeness_centrality_node_22', 'closeness_centrality_node_23', 'closeness_centrality_node_24', 'closeness_centrality_node_25', 'closeness_centrality_node_26', 'closeness_centrality_node_27', 'closeness_centrality_node_28', 'closeness_centrality_node_29', 'closeness_centrality_node_30', 'closeness_centrality_node_31', 'closeness_centrality_node_32', 'closeness_centrality_node_33', 'closeness_centrality_node_34', 'closeness_centrality_node_35', 'closeness_centrality_node_36', 'closeness_centrality_node_37', 'closeness_centrality_node_38', 'closeness_centrality_node_39', 'closeness_centrality_node_40', 'closeness_centrality_node_41', 'closeness_centrality_node_42', 'closeness_centrality_node_43', 'closeness_centrality_node_44', 'closeness_centrality_node_45', 'closeness_centrality_node_46', 'closeness_centrality_node_47', 'closeness_centrality_node_48', 'closeness_centrality_node_49', 'closeness_centrality_node_50', 'closeness_centrality_node_51', 'closeness_centrality_node_52', 'closeness_centrality_node_53', 'closeness_centrality_node_54', 'closeness_centrality_node_55', 'closeness_centrality_node_56', 'closeness_centrality_node_57', 'closeness_centrality_node_58', 'closeness_centrality_node_59', 'closeness_centrality_node_60', 'closeness_centrality_node_61', 'closeness_centrality_node_62', 'closeness_centrality_node_63', 'closeness_centrality_node_64', 'closeness_centrality_node_65', 'closeness_centrality_node_66', 'closeness_centrality_node_67', 'closeness_centrality_node_68', 'closeness_centrality_node_69', 'closeness_centrality_node_70', 'closeness_centrality_node_71', 'closeness_centrality_node_72', 'closeness_centrality_node_73', 'closeness_centrality_node_74', 'closeness_centrality_node_75', 'closeness_centrality_node_76', 'closeness_centrality_node_77', 'closeness_centrality_node_78', 'closeness_centrality_node_79', 'closeness_centrality_node_80', 'closeness_centrality_node_81', 'node_strength_node_0', 'node_strength_node_1', 'node_strength_node_2', 'node_strength_node_3', 'node_strength_node_4', 'node_strength_node_5', 'node_strength_node_6', 'node_strength_node_7', 'node_strength_node_8', 'node_strength_node_9', 'node_strength_node_10', 'node_strength_node_11', 'node_strength_node_12', 'node_strength_node_13', 'node_strength_node_14', 'node_strength_node_15', 'node_strength_node_16', 'node_strength_node_17', 'node_strength_node_18', 'node_strength_node_19', 'node_strength_node_20', 'node_strength_node_21', 'node_strength_node_22', 'node_strength_node_23', 'node_strength_node_24', 'node_strength_node_25', 'node_strength_node_26', 'node_strength_node_27', 'node_strength_node_28', 'node_strength_node_29', 'node_strength_node_30', 'node_strength_node_31', 'node_strength_node_32', 'node_strength_node_33', 'node_strength_node_34', 'node_strength_node_35', 'node_strength_node_36', 'node_strength_node_37', 'node_strength_node_38', 'node_strength_node_39', 'node_strength_node_40', 'node_strength_node_41', 'node_strength_node_42', 'node_strength_node_43', 'node_strength_node_44', 'node_strength_node_45', 'node_strength_node_46', 'node_strength_node_47', 'node_strength_node_48', 'node_strength_node_49', 'node_strength_node_50', 'node_strength_node_51', 'node_strength_node_52', 'node_strength_node_53', 'node_strength_node_54', 'node_strength_node_55', 'node_strength_node_56', 'node_strength_node_57', 'node_strength_node_58', 'node_strength_node_59', 'node_strength_node_60', 'node_strength_node_61', 'node_strength_node_62', 'node_strength_node_63', 'node_strength_node_64', 'node_strength_node_65', 'node_strength_node_66', 'node_strength_node_67', 'node_strength_node_68', 'node_strength_node_69', 'node_strength_node_70', 'node_strength_node_71', 'node_strength_node_72', 'node_strength_node_73', 'node_strength_node_74', 'node_strength_node_75', 'node_strength_node_76', 'node_strength_node_77', 'node_strength_node_78', 'node_strength_node_79', 'node_strength_node_80', 'node_strength_node_81', 'pagerank_node_0', 'pagerank_node_1', 'pagerank_node_2', 'pagerank_node_3', 'pagerank_node_4', 'pagerank_node_5', 'pagerank_node_6', 'pagerank_node_7', 'pagerank_node_8', 'pagerank_node_9', 'pagerank_node_10', 'pagerank_node_11', 'pagerank_node_12', 'pagerank_node_13', 'pagerank_node_14', 'pagerank_node_15', 'pagerank_node_16', 'pagerank_node_17', 'pagerank_node_18', 'pagerank_node_19', 'pagerank_node_20', 'pagerank_node_21', 'pagerank_node_22', 'pagerank_node_23', 'pagerank_node_24', 'pagerank_node_25', 'pagerank_node_26', 'pagerank_node_27', 'pagerank_node_28', 'pagerank_node_29', 'pagerank_node_30', 'pagerank_node_31', 'pagerank_node_32', 'pagerank_node_33', 'pagerank_node_34', 'pagerank_node_35', 'pagerank_node_36', 'pagerank_node_37', 'pagerank_node_38', 'pagerank_node_39', 'pagerank_node_40', 'pagerank_node_41', 'pagerank_node_42', 'pagerank_node_43', 'pagerank_node_44', 'pagerank_node_45', 'pagerank_node_46', 'pagerank_node_47', 'pagerank_node_48', 'pagerank_node_49', 'pagerank_node_50', 'pagerank_node_51', 'pagerank_node_52', 'pagerank_node_53', 'pagerank_node_54', 'pagerank_node_55', 'pagerank_node_56', 'pagerank_node_57', 'pagerank_node_58', 'pagerank_node_59', 'pagerank_node_60', 'pagerank_node_61', 'pagerank_node_62', 'pagerank_node_63', 'pagerank_node_64', 'pagerank_node_65', 'pagerank_node_66', 'pagerank_node_67', 'pagerank_node_68', 'pagerank_node_69', 'pagerank_node_70', 'pagerank_node_71', 'pagerank_node_72', 'pagerank_node_73', 'pagerank_node_74', 'pagerank_node_75', 'pagerank_node_76', 'pagerank_node_77', 'pagerank_node_78', 'pagerank_node_79', 'pagerank_node_80', 'pagerank_node_81']
        #microstructure_cols = ['Subject ID', 'DX_le', 'RG_le', 'GM_Volume', 'WM_Volume', 'BPV', 'mean_MD', 'mean_FA', 'TBSS_WMmaskFA', 'LH_meanMD', 'RH_meanMD', 'LH_medianMD', 'RH_medianMD', 'Left-Thalamus', 'Left-Caudate', 'Left-Putamen', 'Left-Pallidum', 'Left-Hippocampus', 'Left-Amygdala', 'Left-Accumbens-area', 'sum_Hippocampus', 'Right-Thalamus', 'Right-Caudate', 'Right-Putamen', 'Right-Pallidum', 'Right-Hippocampus', 'Right-Amygdala', 'Right-Accumbens-area', 'CSF', 'mean_inferiortemporal_thickness', 'mean_middletemporal_thickness', 'mean_temporalpole_thickness', 'mean_superiorfrontal_thickness', 'mean_superiorparietal_thickness', 'mean_supramarginal_thickness', 'mean_precuneus_thickness', 'mean_superiortemporal_thickness', 'mean_inferiorparietal_thickness', 'mean_rostralmiddlefrontal_thickness']
        microstructure_cols = ['Subject ID', 'DX_le', 'RG_le', 'BPV', 'mean_MD', 'mean_FA', 'TBSS_WMmaskFA', 'LH_meanMD', 'RH_meanMD', 'LH_medianMD', 'RH_medianMD', 'Left-Thalamus', 'Left-Caudate', 'Left-Putamen', 'Left-Pallidum', 'Left-Hippocampus', 'Left-Amygdala', 'Left-Accumbens-area', 'sum_Hippocampus', 'Right-Thalamus', 'Right-Caudate', 'Right-Putamen', 'Right-Pallidum', 'Right-Hippocampus', 'Right-Amygdala', 'Right-Accumbens-area', 'CSF', 'mean_inferiortemporal_thickness', 'mean_middletemporal_thickness', 'mean_temporalpole_thickness', 'mean_superiorfrontal_thickness', 'mean_superiorparietal_thickness', 'mean_supramarginal_thickness', 'mean_precuneus_thickness', 'mean_superiortemporal_thickness', 'mean_inferiorparietal_thickness', 'mean_rostralmiddlefrontal_thickness']

        # Create DataFrames for each category
        demographic_df = dataframe[demographic_cols]
        clinical_df = dataframe[clinical_cols]
        graphtheory_df = dataframe[graphtheory_cols]
        microstructure_df = dataframe[microstructure_cols]

        # Select the specific columns you want from each DataFrame
        combined_df = pd.concat([demographic_df, clinical_df, graphtheory_df, microstructure_df], axis=1)   # Include specific demographic_df_le['Age'] or Exclude specific columns from clinical_df: clinical_df.drop(columns=['column_to_exclude1'])
        
        logging.info(f"Shape of Demographic df: {demographic_df.shape}")
        logging.info(f"Shape of Clinical df: {clinical_df.shape}")
        logging.info(f"Shape of Graph Theory df: {graphtheory_df.shape}")
        logging.info(f"Shape of Microstructure df: {microstructure_df.shape}")
        logging.info(f"Shape of Combined df: {combined_df.shape}")  
        logging.info(f"Columns in Combined df: {list(combined_df.columns)}") 

#####        
        logging.info(f"\n\n############### REMOVING UNNECESSARY COLUMNS AFTER LABEL ENCODING #############")
        target_column = 'DX_le'
        #target_column = 'RG_le'
        y = dataframe[target_column]

        # Drop unnecessary columns if they exist in the DataFrame
        demographic_general_cols = ['Subject ID', 'RG', 'RG_le', 'DX', 'DX_le', 'deltaMMSE/Years', 'VISCODE', 'Sex', 'PTMARRY']
        demographic_cols_to_include = ['Age', 'PTEDUCAT', 'Sex_le', 'PTMARRY_le']
        #demographic_cols_filter_le = ['Sex_le', 'PTMARRY_le']
        #demographic_cols_filter_ohe = ['Sex_M', 'PTMARRY_Widowed', 'PTMARRY_Unknown', 'PTMARRY_Never married', 'PTMARRY_Married']
        
        # Combine All Column Lists to filter-Out (clinical_cols | graphtheory_cols | microstructure_cols)
        # Only MS Metrics
        columns_to_filterout = demographic_general_cols + demographic_cols_to_include + clinical_cols + graphtheory_cols 
        # Only GT Metrics
        #columns_to_filterout = demographic_general_cols + demographic_cols_to_include + clinical_cols + microstructure_cols 
        # Demographic + MS Metrics
        columns_to_filterout = demographic_general_cols + clinical_cols + graphtheory_cols 
        # Age _ PTEDUCAT + GT Metrics
        #columns_to_filterout = demographic_general_cols + demographic_cols_to_include + clinical_cols + microstructure_cols 
        
        # Identify Columns Present in the DataFrame
        columns_to_drop = [col for col in columns_to_filterout if col in combined_df.columns]

        if columns_to_drop:
            combined_df = combined_df.drop(columns=columns_to_drop)
            logging.info(f"Dropped unnecessary columns: {columns_to_drop}")
        else:
            logging.info(f"No unnecessary columns to drop from {columns_to_filterout}")


        logging.info("\n\nPerforming basic EDA after removing unnecessary columns.")
        logging.info(f"Shape of the dataset: {combined_df.shape}")
        logging.info(f"Columns in the dataset: {list(combined_df.columns)}")
        logging.info(f"First few rows of the dataset:\n{combined_df.head()}")
        #logging.info(f"Missing values in the dataset:\n{combined_df.isnull().sum()}")


        # Concatenating the target column back into the dataframe
        dataframe = pd.concat([combined_df, y], axis=1)
        # Save the data after Label Encoding and removing unnecessary columns
        save_data(dataframe, os.path.join(project_output_dir, 'ADNI_CTCKGT_mmse_final.csv'))
    
    else:
        logging.info(f"################################################################################")
        logging.info(f"########### Data already preprocessed. Skipping the preprocessing steps. #######")
        logging.info(f"################################################################################")
        
        # Load preprocessed data for subsequent comparisons
        dataframe = pd.read_csv(os.path.join(project_output_dir, 'ADNI_CTCKGT_mmse_final.csv'))
        logging.info("Loaded preprocessed data for further processing.")

#####
    target_column = 'DX_le'
    #target_column = 'RG_le'
    # Print class distribution before any modifications
    class_distribution = dataframe[target_column].value_counts()
    print(f"Class distribution before processing:\n{class_distribution}")

    # Modify target labels based on classification type and comparison
    if classification_type == 'binary':
        if comparison == 'CN_vs_MCI':
            dataframe = dataframe[dataframe[target_column].isin([0, 1])]
            dataframe[target_column] = dataframe[target_column].map({0: 0, 1: 1})
            logging.info("Performing binary classification: 0 = CN, 1 = MCI")
        elif comparison == 'CN_vs_AD':
            dataframe = dataframe[dataframe[target_column].isin([0, 2])]
            dataframe[target_column] = dataframe[target_column].map({0: 0, 2: 1})
            logging.info("Performing binary classification: 0 = CN, 1 = AD")
        elif comparison == 'MCI_vs_AD':
            dataframe = dataframe[dataframe[target_column].isin([1, 2])]
            dataframe[target_column] = dataframe[target_column].map({1: 0, 2: 1})
            logging.info("Performing binary classification: 0 = MCI, 1 = AD")
    elif classification_type == 'three_level':
        logging.info("Three-level classification: CN vs MCI vs AD")
    
    # Print class distribution after selecting classification_type
    class_distribution = dataframe[target_column].value_counts()
    print(f"Class distribution after choosing {classification_type} classification:\n{class_distribution}")
    

    logging.info(f"\n\n##################### TRAIN-TEST SPLIT and STANDARDIZATION ####################")
    # Identify numeric columns for standardization
    # If required - Convert multiple columns to 'category' type
    columns_to_convert = [target_column]
    #columns_to_convert = ['APOE4', target_column] # if 'APOE4' included and to consider it as cateogorical feature
    dataframe[columns_to_convert] = dataframe[columns_to_convert].astype('category')

    categorical_cols = dataframe.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
    numerical_cols = dataframe.select_dtypes(include=['int64', 'float64']).columns.tolist()
    logging.info(f"List of Categorical Columns: {list(categorical_cols)}")
    logging.info(f"List of Numerical Columns: {list(numerical_cols)}")


    # Extract features and target
    y = dataframe[target_column]
    X = dataframe.drop(target_column, axis=1)
    print("Shape of Features: ", X.shape)
    print("Shape of Target: ", y.shape)

    # Split the data into training and test sets
    logging.info(f"Splitting data into training and test sets with test size {test_size} and seed {seed}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


    # Apply SMOTE to balance the training data if required
    if use_smote:
        logging.info("Applying SMOTE to balance training data")
        smote = SMOTE(random_state=seed)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logging.info(f"Class distribution after SMOTE: \n{y_train.value_counts()}")

    
    # Scale the numeric features on training data and apply the same transformation to the test data
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    
    logging.info(f"Scaling numeric features using {scaler_type} scaler")
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    print("X_train: \n", X_train.head())
    print("\nX_Test: \n", X_test.head())

    logging.info(f"Preprocessed and split data saved to {data_dir}")
    save_data(X_train, os.path.join(data_dir, 'data', 'X_train.csv'))
    save_data(X_test, os.path.join(data_dir, 'data', 'X_test.csv'))
    save_data(y_train, os.path.join(data_dir, 'data', 'y_train.csv'))
    save_data(y_test, os.path.join(data_dir, 'data', 'y_test.csv'))
    logging.info(f"X_train shape: {X_train.shape}")
    logging.info(f"X_test shape: {X_test.shape}")
    logging.info(f"y_train shape: {y_train.shape}")
    logging.info(f"y_test shape: {y_test.shape}")

    logging.info(f"Processed data saved in {data_dir}")
    

def create_directory_structure(output_dir, classification_type, comparison):
    if classification_type == 'binary':
        folder_name = f"{comparison.replace('vs_', '')}"
    else:
        folder_name = 'CN_MCI_AD'

    # Create main directory for classification type
    classification_dir = os.path.join(output_dir, folder_name)
    os.makedirs(os.path.join(classification_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(classification_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(classification_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(classification_dir, 'results'), exist_ok=True)

    return classification_dir


def parse_arguments():
    parser = argparse.ArgumentParser(description="Preprocess the ADNI Data...")
    parser.add_argument('--path', type=str, default="/Users/shahzadali/Documents/Datasets/ADNI", help="Directory path for clinical data files")
    parser.add_argument('--file_name', type=str, default='071124_AD_data.xlsx', help="Name of the clinical data Excel file")  
    parser.add_argument('--sheet_ADNI_preprocessed', type=str, default='ADNI_CMSGT_mmse', help="Sheet name in the Excel file")   # ADNI_CMSGT_mmse_BlncDX | ADNI_CMSGT_mmse_BlncRG
    
    parser.add_argument('--output_dir', type=str, default='Project_Output_combined', help="Main project directory for output data")
    
    parser.add_argument('--scaler', type=str, default='minmax', choices=['standard', 'minmax'], help="Type of scaler to use")
    parser.add_argument('--test_size', type=float, default=0.20, help="Proportion of the dataset to include in the test split")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for data shuffling and splitting")
    parser.add_argument('--use_smote', action='store_true', help="Apply SMOTE for balancing classes in the training data")

    return parser.parse_args()

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = parse_arguments()

    # Construct the file path
    file_path = os.path.join(args.path, args.file_name)

    # Read the data from the specified sheet
    df = pd.read_excel(file_path, sheet_name=args.sheet_ADNI_preprocessed)

    # Define all classification types and comparisons
    CLASSIFICATION_COMPARISONS = {
        'binary': ['CN_vs_AD', 'CN_vs_MCI', 'MCI_vs_AD'],
        'three_level': ['CN_MCI_AD']
    }

    # Perform full preprocessing only on the first run
    first_comparison = True
    for classification_type, comparisons in CLASSIFICATION_COMPARISONS.items():
        for comparison in comparisons:
            # Create directory structure for current comparison
            classification_dir = create_directory_structure(args.output_dir, classification_type, comparison)

            use_smote = False  # Change to False if you don't want to use SMOTE
            
            # Preprocess data for the current classification type and comparison
            preprocess_data(dataframe=df,
                            classification_type=classification_type, 
                            comparison=comparison, 
                            scaler_type=args.scaler, 
                            project_output_dir=args.output_dir, 
                            data_dir=classification_dir,
                            test_size=args.test_size, 
                            seed=args.seed,
                            preprocessed=not first_comparison,
                            use_smote=use_smote
                            )
            first_comparison = False

if __name__ == "__main__":
    main()
