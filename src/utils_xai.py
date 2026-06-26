import pandas as pd
import shap

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
