import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from src.data_selection import (
    select_features,
    remove_low_variance_features,
    perform_univariate_fs,
)


@pytest.fixture
def sample_data():
    # Generate a sample dataset for testing
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    columns = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=columns)
    df["target"] = y
    return df


def test_select_features(sample_data):
    columns_to_keep = ["feature_2", "feature_5", "feature_10"]
    selected_data = select_features(sample_data, columns_to_keep)
    assert isinstance(selected_data, pd.DataFrame)
    assert selected_data.shape[1] == len(columns_to_keep)
    assert all(col in selected_data.columns for col in columns_to_keep)


def test_remove_low_variance_features(sample_data):
    threshold = 0.1
    low_variance_columns = remove_low_variance_features(sample_data, threshold)
    assert isinstance(low_variance_columns, list)
    assert all(col in sample_data.columns for col in low_variance_columns)
    assert len(low_variance_columns) < sample_data.shape[1]


def test_perform_univariate_fs(sample_data):
    target = "target"
    selected_features_list, dropped_features_list = perform_univariate_fs(
        sample_data, target
    )
    assert isinstance(selected_features_list, list)
    assert isinstance(dropped_features_list, list)
    assert all(col in sample_data.columns for col in selected_features_list)
    assert all(col in sample_data.columns for col in dropped_features_list)
    assert len(selected_features_list) == 10
    assert len(dropped_features_list) == sample_data.shape[1] - 1 - 10
