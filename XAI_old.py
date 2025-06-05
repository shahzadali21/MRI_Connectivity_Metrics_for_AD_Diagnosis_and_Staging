

########## OLD ###########
def plot_shap_summary_by_class(model, X_test, class_mapping, output_dir):
    """
    Plot SHAP summary plots for each class in the model.
    """
    shap.initjs()
    
    # Determine which explainer to use based on the model type
    if hasattr(model, 'tree_'):
        explainer = shap.TreeExplainer(model)  # Use TreeExplainer for tree-based models
    elif isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, X_test)  # Use LinearExplainer for linear models
    else:
        explainer = shap.Explainer(model)  # Fallback to generic Explainer


    # Generate SHAP values
    shap_values = explainer.shap_values(X_test)
    #num_classes = len(class_mapping)
    num_classes = shap_values.shape[2]

    for class_index in range(num_classes):
        class_shap_values = shap_values[:, :, class_index]
        shap_df = pd.DataFrame(class_shap_values, columns=X_test.columns)

        mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
        #print(f"Class {class_mapping[class_index]} mean absolute SHAP values:")
        #print(mean_abs_shap)

        plt.figure()
        shap.summary_plot(class_shap_values, X_test, max_display=17, show=False)
        plt.title(f'SHAP Summary Plot - Class {class_mapping[class_index]}')
        plt.savefig(f'{output_dir}/Shap_summary_plot_class_{class_mapping[class_index]}_{model.__class__.__name__}.png', format='png', bbox_inches='tight', dpi=500)
        plt.close()
    print(f"SHAP summary plots for each class is saved to {output_dir}.")








def plot_shap_aggregated_summary(model, X_test, output_dir):
    """
    Plot aggregated SHAP summary plot for the model.
    """
    shap.initjs()
    
    # Determine the appropriate SHAP explainer based on the model type
    if hasattr(model, 'tree_'):
        # Tree-based models (e.g., RandomForest, GradientBoosting, XGBoost)
        explainer = shap.TreeExplainer(model)
    elif isinstance(model, (LogisticRegression, SGDClassifier, LinearDiscriminantAnalysis)):
        # Linear models (e.g., Logistic Regression, SGDClassifier, LDA)
        explainer = shap.LinearExplainer(model, X_test)
    elif isinstance(model, (SVC, GaussianNB, KNeighborsClassifier)):
        # Kernel or non-linear models (e.g., SVM, Naive Bayes, KNN)
        if hasattr(model, 'predict_proba'):
            explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(X_test, min(50, X_test.shape[0])))
        else:
            raise ValueError(f"SVM models without 'probability=True' are not supported. Please enable probability estimates.")
    elif isinstance(model, (XGBClassifier, GradientBoostingClassifier, AdaBoostClassifier)):
        # Boosting models
        explainer = shap.TreeExplainer(model)
    else:
        # Generic models (fallback to default SHAP explainer)
        explainer = shap.Explainer(model, X_test)

    #shap_values = explainer(X_test)
    shap_values = explainer.shap_values(X_test)

    aggregated_shap_values = np.abs(shap_values).mean(axis=2)
    aggregated_shap_df = pd.DataFrame(aggregated_shap_values, columns=X_test.columns)
    mean_abs_aggregated_shap = aggregated_shap_df.abs().mean().sort_values(ascending=False)
    #print(mean_abs_aggregated_shap)

    plt.figure()
    shap.summary_plot(aggregated_shap_values, X_test, max_display=17, show=False)
    plt.title(f'SHAP Summary Plot - Aggregated for all Classes')
    plt.savefig(f'{output_dir}/Shap_summary_plot_Aggregated_{model.__class__.__name__}.png', format='png', bbox_inches='tight', dpi=500)
    plt.close()
    print(f"Aggregated SHAP summary plot is saved to {output_dir}.")



def plot_shap_dependence_and_feature_importance(model, X_test, class_mapping, output_dir, results_dir):
    """
    Plot SHAP dependence plots for all features and compare feature importance across all classes.
    """
    shap.initjs()

    # Determine which explainer to use based on the model type
    if hasattr(model, 'tree_'):
        explainer = shap.TreeExplainer(model)  # Use TreeExplainer for tree-based models
    else:
        explainer = shap.Explainer(model)      # Use general Explainer for other models

    shap_values = explainer.shap_values(X_test)
    num_classes = shap_values.shape[2]

    # Compute feature importance for each class
    feature_importance = pd.DataFrame()
    for class_index in range(num_classes):
        class_shap_values = shap_values[:, :, class_index]
        shap_df = pd.DataFrame(class_shap_values, columns=X_test.columns)

        mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
        feature_importance[f'Class_{class_mapping[class_index]}'] = mean_abs_shap

    # Save the SHAP feature importances to a CSV file with a proper header
    csv_path = os.path.join(results_dir, f'Shap_Feature_Importance_{model.__class__.__name__}.csv')
    feature_importance.index.name = 'Feature'
    feature_importance.to_csv(csv_path)
    print(f"SHAP Feature Importance saved to {csv_path}_{model.__class__.__name__}")
    """
    # Generate dependence plots for each feature and each class
    for class_index in range(num_classes):
        class_shap_values = shap_values[:, :, class_index]
        for feature in X_test.columns:
            plt.figure()
            shap.dependence_plot(feature, class_shap_values, X_test, show=False)
            plt.title(f'Class {class_mapping[class_index]} - {feature}')
            plt.savefig(os.path.join(output_dir, f'Shap_dependence_plot_class_{class_mapping[class_index]}_{feature}.pdf'), format='pdf', bbox_inches='tight')
            plt.close()  # Close the plot to avoid display and memory issues
    print(f"SHAP dependence plots for all features compared to all classes are saved to {output_dir}.")
    """



def plot_force_for_sample(model, X_test, sample_index, output_dir):
    """
    Generate force plots for a specific sample in the test set to determine feature contributions.
    """
    shap.initjs()
    # Determine which explainer to use based on the model type
    if hasattr(model, 'tree_'):
        explainer = shap.TreeExplainer(model)  # Use TreeExplainer for tree-based models
    else:
        explainer = shap.Explainer(model)      # Use general Explainer for other models

    shap_values = explainer.shap_values(X_test)
    num_classes = shap_values.shape[2]

    for class_index in range(num_classes):
        class_shap_values = shap_values[:, :, class_index]
        sample_shap_values = class_shap_values[sample_index, :]

        force_plot = shap.force_plot(explainer.expected_value[class_index], sample_shap_values, X_test.iloc[sample_index, :])  #,matplotlib=True
        
        plt.suptitle(f'Class {class_index} - Force Plot for Sample {sample_index}', y=1.1)        
        plot_filename = f'{output_dir}/Shap_Force_plot_class_{class_index}_sample_{sample_index}_{model.__class__.__name__}.html'
        shap.save_html(plot_filename, force_plot)
        print(f'Force plot saved to {plot_filename}')




def plot_decision_for_sample_and_class(model, X_test, sample_index, class_index, class_mapping, output_dir):
    """
    Generate a decision plot for a specific sample and class.
    """
    shap.initjs()
    # Determine which explainer to use based on the model type
    if hasattr(model, 'tree_'):
        explainer = shap.TreeExplainer(model)  # Use TreeExplainer for tree-based models
    else:
        explainer = shap.Explainer(model)      # Use general Explainer for other models

    shap_values = explainer.shap_values(X_test)

    #class_mapping = {0: 'CN', 1: 'MCI', 2: 'Dementia'}
    class_shap_values = shap_values[:, :, class_index]
    feature_names = X_test.columns.tolist()
    shap.decision_plot(explainer.expected_value[class_index], class_shap_values[sample_index, :], X_test.iloc[sample_index, :].values, feature_names=feature_names, show=False)

    plt.suptitle(f'Class {class_mapping[class_index]} - Decision Plot for Sample {sample_index}', y=1.05)
    plt.savefig(f'{output_dir}/Shap_decision_plot_sample_{sample_index}_class_{class_mapping[class_index]}_{model.__class__.__name__}.png', format='png', bbox_inches='tight', dpi=500)
    plt.close()
    print(f"Shap Decision plot for sample at index {sample_index} and class {class_index} saved to {output_dir}.")




def plot_decision_for_all_samples(model, X_test, class_mapping, output_dir):
    """
    Generate decision plots for all samples for each class.
    """
    shap.initjs()
    # Determine which explainer to use based on the model type
    if hasattr(model, 'tree_'):
        explainer = shap.TreeExplainer(model)  # Use TreeExplainer for tree-based models
    else:
        explainer = shap.Explainer(model)      # Use general Explainer for other models

    shap_values = explainer.shap_values(X_test)
    num_classes = shap_values.shape[2]
    feature_names = X_test.columns.tolist()

    for class_index in range(num_classes): 
        class_shap_values = shap_values[:, :, class_index]
        shap.decision_plot(explainer.expected_value[class_index], class_shap_values, X_test.values, feature_names=feature_names, show=False)
        
        plt.suptitle(f'Class {class_mapping[class_index]} - Decision Plot for All Samples', y=1.05)
        plt.savefig(f'{output_dir}/Shap_Decision_Plot_class_{class_mapping[class_index]}_{model.__class__.__name__}.png', format='png', bbox_inches='tight', dpi=500)
        plt.close()
    print(f"Shap Decision plot for all samples for each class are saved to {output_dir}.")
