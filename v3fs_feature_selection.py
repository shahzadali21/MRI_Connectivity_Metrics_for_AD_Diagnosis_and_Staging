
import logging
import numpy as np

from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.decomposition import PCA

def feature_selection(X_train, y_train, method, k):
    logging.info(f"Performing feature selection using: {method}")
    if method == "none":
        logging.info("Skipping feature selection.")
        selected_features = list(range(X_train.shape[1]))  # Return all feature indices
        return X_train, selected_features
    
    elif method == "mutual_info":
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        selected_features = selector.get_support(indices=True)

    elif method == "anova":
        selector = SelectKBest(score_func=f_regression, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        selected_features = selector.get_support(indices=True)

    elif method == "rfe_elastic_net":
        selector = RFE(estimator=ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42), n_features_to_select=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        selected_features = selector.get_support(indices=True)

    elif method == "random_forest":
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        importances = model.feature_importances_
        selected_features = np.argsort(importances)[-k:]
        X_train_selected = X_train.iloc[:, selected_features]

    elif method == "pca":
        max_components = min(X_train.shape[0], X_train.shape[1])
        if k > max_components:
            logging.warning(f"Requested number of components (k={k}) exceeds max allowed ({max_components}). Reducing k to {max_components}.")
            k = max_components
        pca = PCA(n_components=k)
        X_train_selected = pca.fit_transform(X_train)
        logging.info(f"Explained variance by PCA components: {pca.explained_variance_ratio_}")
        selected_features = list(range(k))

    else:
        raise ValueError(f"Unsupported feature selection method: {method}")

    logging.info(f"Selected features (indices): {selected_features}")
    logging.info(f"Selected features (names): {X_train.columns[selected_features].tolist() if hasattr(X_train, 'columns') else selected_features}")

    return X_train_selected, selected_features
