from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif
from src.utils import flatten
from typing import List, Tuple
import numpy as np
import pandas as pd
import logging


def select_features(
    data: pd.DataFrame,
    columns_to_keep: list,
) -> pd.DataFrame:
    """
    This function performs manual feature selection, if the string with the columns to keep is passed as argument.
    The columns that have not been manually selected accordinlgy to the domain knowledge are dropped.
    Args:
        data (DataFrame): DataFrame with the data.
        columns_to_keep (list): list of columns to keep. They have been manually selected and specified in the merged_parameters.
    Returns:
        selected_data (DataFrame): DataFrame after the column selection.
    """
    if columns_to_keep is not None:
        columns_to_keep = flatten(columns_to_keep.values())
        selected_data = data[columns_to_keep]
    else:
        selected_data = data
    return selected_data


def apply_filter(data, feature_name, value_range):
    """Applies a filter on a specific feature of the DataFrame.
    Args:
        data (DataFrame): DataFrame to be filtered.
        feature_name (str): Name of the feature/column to filter.
        value_range (list): Minimum and maximum values for the feature range, inclusive.
        log_message (str): Description for log message.   
    Returns:
        data (DataFrame): Filtered DataFrame.
        filter_applied (bool): Boolean indicating if filter was applied.
    """
    filter_applied = False

    if value_range is not None and not (value_range[0] is None and value_range[1] is None):
        min_val, max_val = 0, np.Inf
        if value_range[0] is not None:
            min_val = value_range[0]
        if value_range[1] is not None:
            max_val = value_range[1]

        if min_val > max_val:
            logging.warning(f"Invalid {feature_name} range: {min_val} should be less than or equal to {max_val}. No filter applied.")
        else:
            filter_condition = (data[feature_name] >= min_val) & (data[feature_name] <= max_val)
            data = data.loc[filter_condition]
            filter_applied = True
            logging.info(f"{feature_name} filter applied. Range: [{min_val}, {max_val}]")

    return data, filter_applied


def filter_feature(
    data: pd.DataFrame,
    dict_features: dict,
):
    """This function applies filters on age, sex, and BMI features.
    The function filters these features to identify other features that might have a stronger impact on model predictions.
    Args:
        data (DataFrame): DataFrame containing the data to be filtered.
        dict_features (dict): dictionary containing the name of the features to filter and the respective values.
    Returns:
        data (DataFrame): Filtered DataFrame.
    """
    filter_applied = False

    if dict_features is not None:
        if "age" in dict_features:
            data, age_filter = apply_filter(data, *dict_features["age"])
            filter_applied = age_filter

        if "bmi" in dict_features:
            data, bmi_filter = apply_filter(data, *dict_features["bmi"])
            filter_applied = bmi_filter or filter_applied

        if "sex" in dict_features:
            sex = dict_features["sex"][1]
            if sex in ["M", "W", "m", "w"]:
                sex_mapping = {"M": 0, "W": 1, "m": 0, "w": 1}
                filter_sex = data[dict_features["sex"][0]] == sex_mapping[sex]
                data = data.loc[filter_sex]
                logging.info(f"Sex filter applied. Sex: {sex}")
                filter_applied = True
            elif sex is not None:
                logging.warning("Sex filter NOT applied due to an invalid entry.")

    if filter_applied is False:
        logging.info("No filter applied")
    return data


# Extra functions to test feature selection methods, not used so far


def remove_low_variance_features(
    data: pd.DataFrame,
    threshold=0.1,
) -> List[str]:
    """
    This function is a test to see if the columns we deicded to drop have low variance, and thus are not carring interesting informations.
    Args:
        data (DataFrame): DataFrame with the data.
        threshold = threshold for variance
    Returns:
        low_variance_columns (list): list of columns with variance lower than the threshold
    """

    if "g_id_inspire_g_5km" in data.columns:
        data.drop(["g_id_inspire_g_5km"], axis=1)
    if "g_grid_idr_1km" in data.columns:
        data.drop(["g_grid_idr_1km"], axis=1)
    if "basis_udat" in data.columns:
        data.drop(["basis_udat"], axis=1)

    sel = VarianceThreshold(threshold=threshold)
    sel.fit(data)
    low_variance_columns = data.columns[~sel.get_support()].tolist()
    return low_variance_columns


def perform_univariate_fs(
    data: pd.DataFrame,
    target: pd.Series,
) -> Tuple[List[str], List[str]]:
    # data = data.drop(['g_id_inspire_g_5km', 'g_grid_idr_1km', 'basis_udat'], axis=1)
    X = data.drop(
        labels=target,
        axis=1,
    )
    y = data[target]
    selector = SelectKBest(
        f_classif,
        k=10,
    )
    selected_features = selector.fit_transform(
        X=X,
        y=y,
    )
    selected_mask = selector.get_support()
    all_features = X.columns.tolist()
    selected_features_list = [
        feature for feature, mask in zip(all_features, selected_mask) if mask
    ]
    dropped_features_list = [
        feature for feature, mask in zip(all_features, selected_mask) if not mask
    ]

    return (
        selected_features_list,
        dropped_features_list,
    )
