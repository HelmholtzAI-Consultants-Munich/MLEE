from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif
from src.utils import flatten
from typing import Any, List, Tuple
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


def _apply_numerical_filter(data, feature_name, value_range):
    """This function applies a numeric range filter on a specific feature of the DataFrame.
    Args:
        data (DataFrame): DataFrame with the data.
        feature_name (str): Name of the feature to filter.
        value_range (list): List with two elements [min, max] specifying the range to keep for the numeric feature. Use None for no limit.
    Returns:
        data (DataFrame): DataFrame after applying the numeric filter.
        filter_applied (bool): True if the filter was applied, False otherwise.
    """
    filter_applied = False

    if value_range is not None and not (
        value_range[0] is None and value_range[1] is None
    ):
        min_val = value_range[0]
        max_val = value_range[1]

        data_min = data[feature_name].min()
        data_max = data[feature_name].max()

        if min_val is None:
            min_val = data_min
        if max_val is None:
            max_val = data_max

        if min_val > max_val:
            raise ValueError(
                f"Invalid range specified for '{feature_name}': "
                f"the lower bound ({min_val}) cannot be greater than "
                f"the upper bound ({max_val})."
            )

        if min_val < data_min or max_val > data_max:
            raise ValueError(
                f"Invalid {feature_name} range: [{min_val}, {max_val}]. "
                f"Available range is [{data_min}, {data_max}]."
            )

        filter_condition = (
            (data[feature_name] >= min_val)
            & (data[feature_name] <= max_val)
        )
        data = data.loc[filter_condition]
        filter_applied = True
        logging.info(f"{feature_name} filter applied. Range: [{min_val}, {max_val}]")
    else:
        logging.info(f"Filter {feature_name} skipped (no range specified).")

    return data, filter_applied

def _apply_categorical_filter(data, feature_name, filter_values):
    """This function applies a categorical filter on a specific feature of the DataFrame.
    Args:
        data (DataFrame): DataFrame with the data.
        feature_name (str): Name of the feature to filter.
        filter_values (list): List of values to keep for the categorical feature.
    Returns:
        data (DataFrame): DataFrame after applying the categorical filter.
        filter_applied (bool): True if the filter was applied, False otherwise.
    """
    
    filter_condition = data[feature_name].isin(filter_values)
    data = data.loc[filter_condition]
    logging.info(f"{feature_name} filter applied. Categories: {filter_values}")
    filter_applied = True

    return data, filter_applied


def _normalize_filter_definitions(data: pd.DataFrame, dict_features: Any) -> List[dict]:
    """Normalizes filter definitions into a list of dictionaries."""
    if dict_features is None:
        return []

    if isinstance(dict_features, list):
        filter_definitions = dict_features
    elif isinstance(dict_features, dict):
        if any(isinstance(value, dict) for value in dict_features.values()):
            filter_definitions = list(dict_features.values())
        else:
            filter_definitions = []
            for _, filter_value in dict_features.items():
                if isinstance(filter_value, (list, tuple)) and len(filter_value) >= 2:
                    feature_name = filter_value[0]
                    filter_entry = filter_value[1]
                    if isinstance(filter_entry, (list, tuple)) and len(filter_entry) == 2:
                        filter_definitions.append(
                            {
                                "feature": feature_name,
                                "type": "numeric",
                                "range": list(filter_entry),
                            }
                        )
                    else:
                        filter_definitions.append(
                            {
                                "feature": feature_name,
                                "type": "categorical",
                                "value": filter_entry,
                            }
                        )
                else:
                    filter_definitions.append(filter_value)
    else:
        raise TypeError("dict_features must be a dict or list of filter definitions")

    return [definition for definition in filter_definitions if isinstance(definition, dict)]


def _apply_filter_definition(data: pd.DataFrame, filter_definition: dict) -> Tuple[pd.DataFrame, bool]:
    """This function applies a single generic filter definition to the DataFrame.
    Args:
        data (DataFrame): DataFrame with the data.
        filter_definition (dict): Dictionary defining the filter to apply.
    Returns:
        data (DataFrame): DataFrame after applying the filter.
        filter_applied (bool): True if the filter was applied, False otherwise.
    """

    # Check number of istancees before filtering
    n_before = len(data)

    if not isinstance(filter_definition, dict):
        logging.warning("Invalid filter definition. No filter applied.")
        return data, False

    feature_name = filter_definition.get("feature")
    if feature_name is None:
        logging.warning("Filter definition missing a feature name. No filter applied.")
        return data, False

    if feature_name not in data.columns:
        logging.warning(f"Feature {feature_name} not found. No filter applied.")
        return data, False

    filter_type = filter_definition.get("type")

    if filter_type == "numeric":
        value_range = filter_definition.get("range")

        # Check if the range is provided for the numeric feature
        if value_range is None:
            raise ValueError(f"Numeric filter '{feature_name}' must specify a 'range'.")

        # Check if the range is a list or tuple with exactly two elements
        if not isinstance(value_range, (list, tuple)) or len(value_range) != 2:
            raise ValueError(f"Numeric filter '{feature_name}' must specify 'range' as [min, max].")

        data, filter_applied = _apply_numerical_filter(data, feature_name, value_range)

    elif filter_type == "categorical":
        filter_values = filter_definition.get("values")

        # Check if the values are provided for the categorical feature and if they are valid
        if not isinstance(filter_values, (list, tuple)) or len(filter_values) == 0:
            raise ValueError(f"Categorical filter '{feature_name}' must specify 'values' as a non-empty list.")

        # Check if the filter values are valid for the categorical feature
        available_values = data[feature_name].dropna().unique().tolist()
        if not all(value in available_values for value in filter_values):
            raise ValueError(
                f"Invalid values for '{feature_name}'. "
                f"Available values are: {available_values}.")

        data, filter_applied = _apply_categorical_filter(data, feature_name, filter_values)

    else:
        raise ValueError(f"Unsupported filter type '{filter_type}'. Expected 'numeric' or 'categorical'.")

    # Check number of istancees after filtering
    n_after = len(data)
    logging.info(f"Remaining instances after filtering: {n_before} -> {n_after}.")

    return data, filter_applied


def filter_feature(
    data: pd.DataFrame,
    dict_features: Any,
):
    """Applies configurable filters on one or more features.

    The function accepts either the legacy dictionary format used before
    (for example {'age': ['feature_name', [min, max]]}) or a list of generic
    filter definitions such as [{'feature': 'age', 'type': 'numeric', 'range': [20, 40]}].
    Up to four filters can be applied in sequence.
    """
    filter_applied = False

    if dict_features is not None:
        filter_definitions = _normalize_filter_definitions(data, dict_features)
        if len(filter_definitions) > 4:
            raise ValueError("A maximum of four filters can be applied.")

        for filter_definition in filter_definitions:
            data, is_filter_applied = _apply_filter_definition(data, filter_definition)
            filter_applied = filter_applied or is_filter_applied

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
