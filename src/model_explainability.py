import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
import shap
from pathlib import Path
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

#---- Permutation feature importance ----#
def plot_permutation_feature_importance(
    result: dict,
    data: pd.DataFrame,
    title: str,
    importance_plot: str,
) -> None:
    # Sort features by mean importance (ascending for horizontal barh)
    perm_sorted_idx = result.importances_mean.argsort()
    perm_indices = np.arange(0, len(result.importances_mean)) + 0.5

    # Get mean and std 
    means_sorted = result.importances_mean[perm_sorted_idx]
    stds_sorted = result.importances_std[perm_sorted_idx]

    fig, (ax1, ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(15, 8),
    )
    fig.suptitle(title)

    # Plot horizontal bars with error bars showing standard deviation
    ax1.barh(
        perm_indices,
        means_sorted,
        xerr=stds_sorted,
        height=0.7,
        color="#3470a3",
        ecolor="#2c3e50",
        error_kw={"capsize": 4},
    )
    ax1.set_yticks(perm_indices)
    ax1.set_yticklabels(data.columns[perm_sorted_idx])
    ax1.set_ylim((0, len(result.importances_mean)))
    ax1.set_xlabel("Mean permutation importance")

    # Boxplot of the permutation importances (distribution over repeats)
    ax2.boxplot(
        result.importances[perm_sorted_idx].T,
        vert=False,
        labels=data.columns[perm_sorted_idx],
    )

    # Ensure output directory exists; support both str and Path inputs
    path_figures_importance = Path(importance_plot).parent
    os.makedirs(path_figures_importance, exist_ok=True)

    plt.tight_layout()
    fig.savefig(str(importance_plot))
    # plt.show()


def perform_permutation_feaure_importance(
    classifiers: list,
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str,
    output_dir: str,
) -> dict:
    """This function computs feature importance for the model selected by the GridSearchCV and on the validation set
    Args:
        classifiers (list): list of tuned classifiers obtained from the grid search
        X (DataFrame): dataset with all the features used for training and inference
        y (series): target variable
        best_model ():
        dataset_name (str): dataset name to use for title, possible options: train, test, validation
        output_dir (str): name of output directory for reporting
    Returns:
        importance_df (dict): table with mean importance value for each feature
    """

    importance_df_dict = {}
    for classifier in classifiers:
        model_name = list(classifier["classifier"].named_steps.keys())[-1]
        result = permutation_importance(
            estimator=classifier["classifier"],
            X=X,
            y=y,
            n_repeats=30,
            random_state=42,
        )

        # Get feature importances
        importances = result.importances_mean
        importances_std = result.importances_std

        # Create a DataFrame to display feature importances
        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Mean Importance": importances,
            "Standard Deviation Importance": importances_std,
        })

        # Sort by mean importance descending
        importance_df.sort_values(by="Mean Importance", ascending=False, inplace=True)

        # Convert to a tidy dict format that the template expects
        tidy = {
            "features": importance_df["Feature"].tolist(),
            "importances": importance_df["Mean Importance"].tolist(),
            "std": importance_df["Standard Deviation Importance"].tolist(),
        }

        importance_df_dict[model_name] = tidy
        logging.info("Importance - %s", dataset_name)
        logging.info(tidy)

        # Plot feature importances
        title = (
            "Permutation Feature Importance with Variance - "
            + model_name
            + " - "
            + dataset_name
            + "_set"
        )

        importance_plot = (
            output_dir
            + "/importance/"
            + "/"
            + model_name
            + "/"
            + model_name
            + "_"
            + dataset_name
            + "_importance.png"
        )
        plot_permutation_feature_importance(
            result=result,
            data=X,
            title=title,
            importance_plot=importance_plot,
        )

    return importance_df_dict

#---- SHAP----#
def perform_shap(
    models: list,
    classifiers: list,
    best_model_name,
    X: pd.DataFrame,
    X_before_preprocessing: pd.DataFrame,
    y: pd.Series,
    shap_output_prob: bool,
    dataset_name: str,
    output_dir: str,
) -> None:
    """This function computes Shapley values for the model selected from the evaluation step (on validation set).
    For now the only two possible models for which this is computed are Random Forest and XGBoost.
    Args:
        models (list): list of models defined from the model class 
        classifiers (list): list of tuned classifiers obtained from the grid search
        best_model_name (str): name of the best classifier selected from the validation set
        X (DataFrame): dataset with all the features used for training and inference
        y (series): target variable
        shap_output (str): output type for SHAP values, possible options: "ptrobability", "logit"
        dataset_name (str): dataset name to use for title, possible options: train, test, validation
        output_dir (str): name of output directory for reporting
    Returns:
        None
    Notes:
        This fucntion plots global plots for the SHAP values, in particular:
        - global bar plot
        - beeswarm plot
        - violin plot
    """

    for model, classifier in zip(models, classifiers):
        model_name = list(classifier["classifier"].named_steps.keys())[-1]

        # Only perform SHAP for the best model
        if model_name != best_model_name:
            continue

        importance_plot = Path(
            output_dir + "/importance/" + model_name + "_" + dataset_name + "_"
        )
        path_figures_importance = importance_plot.parent
        os.makedirs(
            path_figures_importance,
            exist_ok=True,
        )
        explainer=None
        if best_model_name in ("XGBoost", "rfc"):
            data=None
            feature_perturbation="tree_path_dependent"
            shap_output="raw"

            if shap_output_prob:
                background=shap.sample(X, nsamples=200, random_state=42)
                data=background
                shap_output="probability"
                feature_perturbation="interventional"
                print("Computing SHAP in probability space with interventional perturbation")
            else:
                print("Computing SHAP in raw output space with tree_path_dependent perturbation")       
            
            explainer = shap.TreeExplainer(
                classifier["classifier"][model_name],
                data=data,
                model_output=shap_output,
                feature_perturbation=feature_perturbation,
            )

        if explainer is None: 
            continue

        # Compute SHAP
        shap_values_raw = explainer(X)

        values = shap_values_raw.values
        base_values = shap_values_raw.base_values

        # Handle multi-output (RF probability or multiclass)
        if values.ndim == 3:
            values = values[:, :, 1]

            if isinstance(base_values, np.ndarray) and base_values.ndim > 1:
                base_values = base_values[:, 1]

        shap_values = shap.Explanation(
            values=values,
            base_values=base_values,
            data=X,
            feature_names=shap_values_raw.feature_names
        )

        shap_values_aggregated = aggregate_SHAP_importance_values(
            shap_values=shap_values,
            X_before_preprocessing=X_before_preprocessing
        )

        # Ensure the output directory exists
        output_dir_model = os.path.join(path_figures_importance, model_name)
        os.makedirs(output_dir_model, exist_ok=True)

        # Save the global plots to files
        # XGBoost
        #if best_model_name == "XGBoost":
        print(f"Performing SHAP for {best_model_name}")
        plt.figure()
        shap.summary_plot(
            shap_values_aggregated,
            plot_type="bar",
            show=False,
            max_display=20
        )
        plt.xlabel("mean(|SHAP|)")
        plt.savefig(
            os.path.join(
                output_dir_model, f"{model_name}_shap_bar_{dataset_name}.png"
            )
        )
        plt.close()

        plt.figure()
        shap.plots.violin(
            shap_values_aggregated,
            max_display=20,
            show=False
        )
        plt.tight_layout()
        plt.xlabel("SHAP value (impact on model output)")
        plt.savefig(
            os.path.join(
                output_dir_model, f"{model_name}_shap_violin_{dataset_name}.png"
            )
        )
        plt.close()

        plt.figure()
        shap.plots.beeswarm(
            shap_values_aggregated,
            max_display=20,
            show=False
        )
        plt.tight_layout()
        plt.xlabel("SHAP value (impact on model output)")
        plt.savefig(
            os.path.join(
                output_dir_model, f"{model_name}_shap_beeswarm_{dataset_name}.png"
            )
        )
        plt.close()

        # # rfc
        # if best_model_name == "rfc":
        #     print("Performing SHAP for rfc")
        #     classes = [0, 1]
        #     for c in classes:
        #         plt.figure()
        #         shap.summary_plot(shap_values[:, :, c], X, plot_type="bar", show=False, max_display=20) # max_display=len(X.columns) to show all features in the plot
        #         plt.xlabel("mean(|SHAP|)")
        #         plt.savefig(
        #             os.path.join(
        #                 output_dir_model, f"{model_name}_shap_bar_{dataset_name}_class_{c}.png"
        #             )
        #         )
        #         plt.close()

        #         plt.figure()
        #         shap.summary_plot(
        #             shap_values[:, :, c], X, plot_type="violin", show=False
        #         )
        #         plt.xlabel("SHAP value")
        #         plt.savefig(
        #             os.path.join(
        #                 output_dir_model, f"{model_name}_shap_violin_{dataset_name}_class_{c}.png"
        #             )
        #         )
        #         plt.close()

        #         plt.figure()
        #         shap.summary_plot(shap_values[:, :, c], X, plot_type="dot", show=False)
        #         plt.xlabel("SHAP value")
        #         plt.savefig(
        #             os.path.join(
        #                 output_dir_model, f"{model_name}_shap_beeswarm_{dataset_name}_class_{c}.png"
        #             )
        #         )
        #         plt.close()

def aggregate_SHAP_importance_values(
    shap_values: pd.DataFrame,
    X_before_preprocessing: pd.DataFrame,
) -> shap.Explanation:
    """This function aggregates SHAP values across multi class categorical features
    by taking the mean of the absolute SHAP values across the different classes
    for each multiclass categorical feature and renames the columns to match the original feature names before preprocessing.
    Args:
        - shap_values: a SHAP values object computed from perform_shap function
        - X_before_preprocessing: the original dataframe before preprocessing, used to get the original column names and the index for the new SHAP explanation object
    Returns:
        - grouped_expl: a SHAP explanation object with the aggregated SHAP values and the original column names before preprocessing
    """
    column_list_new = []
    column_list = shap_values.feature_names
    # Retrieve the original column names before preprocessing by splitting the feature names
    # and taking the last part
    for i in column_list:
        if i.startswith(("num__", "bin_cat__")):
            column_list_new.append(i.split("__")[-1])
        else:
            i = i.split("__")[-1]
            base, last = i.rsplit("_", 1)
            column_list_new.append(base)
    # Create a new dataframe with the SHAP values and the original column names before preprocessing
    df_shap_enc = pd.DataFrame(
        shap_values.values,
        columns=shap_values.feature_names,
        index=X_before_preprocessing.index)
    # Group the SHAP values by the original column names before preprocessing
    df_renamed = df_shap_enc.copy()
    df_renamed.columns = column_list_new
    df_renamed = df_renamed.groupby(level=0, axis=1).sum()
    grouped_cols = df_renamed.columns.tolist()
    X_grouped = X_before_preprocessing.loc[:, grouped_cols]
    # Convert dataframe in an explanation SHAP object,
    grouped_expl = shap.Explanation(
    values=df_renamed.values,
    base_values=shap_values.base_values,
    data=X_grouped.values, 
    feature_names=grouped_cols
    )

    return grouped_expl
