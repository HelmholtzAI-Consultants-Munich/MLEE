from sklearn.model_selection import train_test_split
import pandas as pd
import logging
import typing


def train_test_split_stratified(
    data: pd.DataFrame,
    target: str,
    feature_stratification: list,
    test_size: float,
    validation_size: float,
    random_state=42,
) -> typing.Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """
    This function performs a stratified train-test split on the data based on the specified feature stratification.
    Args:
        data (DataFrame): input data.
        target (str): name of the target column in `data`.
        feature_stratification (list): list of feature columns to be used for stratification.
        test_size (float): The proportion of the dataset to include in the test split, defined in the input parameters.
        validation_size (float): The proportion of the dataset to include in the validation split, defined in the input parameters.
    Returns:
        X_train (DataFrame): Training features.
        X_test (DataFrame): Testing features.
        X_validation (DataFrame): Testing features.
        y_train (Series or array-like): Training target variable.
        y_test (Series or array-like): Testing target variable.
        y_validation (Series or array-like): Testing target variable.
    Notes:
        - Assumes X and y have compatible shapes.
        - Stratifies the train-test split based on the specified feature stratification.
        - If feature_stratification is None or an empty list, performs a non-stratified train-test split.
    """
    # `target` is the column name (string) in `data`; drop it to form X and extract y
    X = data.drop(labels=target, axis=1)
    y = data[target]
    test_and_val_size = test_size + validation_size
    percentage_val_size = validation_size / test_and_val_size

    if feature_stratification is not None:
        logging.info("Performed stratified train-test-validation split.")
        stratification_mask = data[feature_stratification]
        X_train, X_test_temp, y_train, y_test_temp = train_test_split(
            X,
            y,
            stratify=stratification_mask,
            test_size=test_and_val_size,
            random_state=random_state,
        )
        data_temp = pd.DataFrame.copy(X_test_temp)
        data_temp[target] = y_test_temp
        X_test, X_val, y_test, y_val = train_test_split(
            X_test_temp,
            y_test_temp,
            stratify=data_temp[feature_stratification],
            test_size=percentage_val_size,
            random_state=random_state,
        )
    else:
        logging.info("Performed train-test-validation split.")
        X_train, X_test_temp, y_train, y_test_temp = train_test_split(
            X,
            y,
            stratify=None,
            test_size=test_and_val_size,
            random_state=random_state,
        )
        X_test, X_val, y_test, y_val = train_test_split(
            X_test_temp,
            y_test_temp,
            stratify=None,
            test_size=percentage_val_size,
            random_state=random_state,
        )

    return (
        X_train,
        X_test,
        X_val,
        y_train,
        y_test,
        y_val,
    )
