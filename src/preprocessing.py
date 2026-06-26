from typing import Tuple
import pandas as pd
import os
import logging
import matplotlib
import matplotlib.pyplot as plt
from builtins import FileNotFoundError
import miceforest as mf

# sklearn imports for safe in-CV preprocessing
from sklearn.experimental import enable_iterative_imputer # noqa: F401
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
try:
    from missingpy import MissForest
except Exception:
    MissForest = None

try:
    from sklearn.impute import KNNImputer
except Exception:
    KNNImputer = None

# TODO: include other imputation methods will be done soon, bit for the moment it't not the priority
# from miceforest import mean_match_shap
# import sklearn.neighbors._base  # TODO private functions should usually not be imported and used
# import sys
# sys.modules["sklearn.neighbors.base"] = sklearn.neighbors._base
# from sklearn.impute import KNNImputer
# from missingpy import MissForest

# Preprocessing (simple imputation and one hot encoding)
# Create a function that handles missing values that I can use in grid search CV, and includes one hot encoding

# Missing values handling
# Deletion
def delete_missing_values(
    data: pd.DataFrame,
    target: str,
    list_columns: list,
) -> pd.DataFrame:
    """
    This function handles the missing values
    Args:
        data (DataFrame):  input data
        target (str): name of the feature that will be the target variable
        list_columns (list): list of string with the names of the columns with MCAR
    Returs:
        data (DataFrame): data with missing values handled
    Notes:
        Rows with missing values in the target feature column will be dropped.
    """
    # Check for missing values
    missing_values = data.isnull().sum()

    if missing_values.sum() == 0:
        logging.info("No missing values found.")
        return data

    # # Drop features that are MCAR accordingly to the chi-square test
    original_data_size = data.shape[0]
    data = data.dropna(subset=list_columns, axis=0)
    # logging.info(
    #     "Drop istances with missing values in the MCAR case. Size of the dataset after dropping: %s",
    #     str(data.shape),
    # )
    # logging.info(
    #     "Percentage of deleted data: %.2f %%",
    #     (original_data_size - final_data_size) / original_data_size * 100.0,
    # )
    # Drop all rows that have a missing value in the target variable
    if target:
        if data[target].isnull().sum():
            data = data.dropna(subset=[target], axis=0)  # , inplace=True)
            final_data_size = data.shape[0]
            percentage_deleted = (
                (original_data_size - final_data_size) / original_data_size * 100.0
            )
            logging.info(
                "The target variable contains missing values. The respective istances are dropped."
            )
            logging.info("Percentage of dropped istances = %.2f %%", percentage_deleted)
            logging.info("Size of the dataset after dropping: %s", str(data.shape))
    return data

# Imputation
def impute_with_univariate(
    data: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """
    This function handles the missing values
    Args:
        data (DataFrame):  input data
        parameters (dict): dictionary with all the input paramters
    Returs:
        data (DataFrame): data with missing values handled
    Notes:
        Rows with missing values in the target feature column will be dropped.
    """
    # Handle missing values
    numerical_columns = parameters["columns_to_keep"]["num"]
    categorical_columns = parameters["columns_to_keep"]["cat"]

    # Replace missing numeric values with mean
    logging.info("Numerical columns: %s", str(numerical_columns))
    logging.info("Number of numerical columnss: %s", str(len(numerical_columns)))
    data.loc[:, numerical_columns] = data.loc[:, numerical_columns].fillna(
        data.loc[:, numerical_columns].mean()
    )

    # Replace missing categorical values with most frequent value
    # Remove the target column from categorical_columns if present
    target_col = None
    if isinstance(parameters, dict):
        target_col = parameters.get("target")
    if target_col:
        categorical_columns = [item for item in categorical_columns if item != target_col]
    logging.info("Categorical columns: %s", str(categorical_columns))
    logging.info("Number of categorical columnss: %s", str(len(categorical_columns)))
    data.loc[:, categorical_columns] = data.loc[:, categorical_columns].fillna(
        data[categorical_columns].mode().iloc[0]
    )
    logging.info(
        "Missing values handled: data imputation with mean (numerical) and most frequent value (categorical)."
    )
    return data


def impute_with_miceforest(
    data: pd.DataFrame,
    imputation_plot: str,
    datasets=1,
) -> Tuple[pd.DataFrame, object]:
    """This function impute missing values with the miceforest package
    Args:
        data (DattaFrame): input data
    Returns:
        inputed_data (DataFrame): dataframe after imputation
    """
    # Set the Matplotlib backend to 'Agg' to avoid interactive plotting
    matplotlib.use("Agg")

    # Create kernel.
    kds = mf.ImputationKernel(
        data,
        datasets=datasets,
        save_all_iterations=True,
        random_state=42,
        # mean_match_scheme=mean_match_shap
    )

    # Run the MICE algorithm for 4 iterations
    kds.mice(4)

    # Return the completed dataset.
    imputed_data = kds.complete_data()

    # Diagnostic plots
    path_figures_imputation = imputation_plot.parent
    os.makedirs(path_figures_imputation, exist_ok=True)
    kds.plot_imputed_distributions(wspace=0.1, hspace=0.2)
    plt.tight_layout()
    plt.savefig(imputation_plot)

    if datasets > 1:
        fig = plt.figure(figsize=(20, 15))
        kds.plot_feature_importance(
            dataset=0, annot=True, cmap="YlGnBu", vmin=0, vmax=1
        )
        plt.tight_layout()
        # TODO change this figure creation if it is actually used
        plt.savefig(path_figures_imputation + "/feature_importance.png")
        print(data.columns)
        # plt.figure(figsize=(20, 15))
        kds.plot_correlations(variables=data.columns)
        plt.tight_layout()
        plt.savefig(path_figures_imputation + "/correlations.png")
        logging.info("Imputation with MICEforest completed!")
    return imputed_data, kds


def impute_with_missforest(
    data: pd.DataFrame,
):
    """
    TODO:include R package - do not consider for now.
    Impute missing values in the DataFrame using missrandom package.
    Args:
        data (DataFrame): Input DataFrame with missing values.
    Returns:
        imputed_data (DataFrame): DataFrame with missing values imputed.
    """

    # # Check if the target_column has missing values
    # if data.isnull().sum() == 0:
    #     print("No missing values to impute.")
    #     return data

    # Perform imputation with missrandom
    imputer = MissForest(
        criterion="absolute_error", max_features="sqrt", oob_score=True, verbose=False
    )
    imputed_data = imputer.fit_transform(data)

    return imputed_data


def impute_with_KNN(
    data: pd.DataFrame,
):
    """TODO: decide to include or not - not consider it now"""
    imputer = KNNImputer(n_neighbors=2, weights="uniform")
    imputed_data = imputer.fit_transform(data)
    return imputed_data


def process_imputation(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_val: pd.DataFrame,
    parameters: dict,
    output_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    TODO: descirbe function"""
    logging.info("Imputation technique: %s", parameters["imputation"])
    if parameters["imputation"] == "univariate":
        X_train = impute_with_univariate(X_train, parameters)
        X_val = impute_with_univariate(X_val, parameters)
        X_test = impute_with_univariate(X_test, parameters)
    elif parameters["imputation"] == "miceforest":
        X_train, _ = impute_with_miceforest(
            data=X_train,
            imputation_plot=output_dir + "/imputation/train_imputation.png",
            datasets=1,
        )
        X_val, _ = impute_with_miceforest(
            data=X_val,
            imputation_plot=output_dir + "/imputation/val_imputation.png",
            datasets=1,
        )
        X_test, _ = impute_with_miceforest(
            data=X_test,
            imputation_plot=output_dir + "/imputation/test_imputation.png",
            datasets=1,
        )
    elif parameters["imputation"] == "missforest":
        X_train = impute_with_missforest(X_train)
        X_val = impute_with_missforest(X_val)
        X_test = impute_with_missforest(X_test)
    elif parameters["imputation"] == "KNN":
        X_train = impute_with_KNN(X_train)
        X_val = impute_with_KNN(X_val)
        X_test = impute_with_KNN(X_test)
    elif parameters["imputation"] == None:
        X_train = delete_missing_values(X_train, None, parameters, X_train.columns)
        X_val = delete_missing_values(X_val, None, parameters, X_val.columns)
        X_test = delete_missing_values(X_test, None, parameters, X_test.columns)

    return (
        X_train,
        X_test,
        X_val,
    )


def make_preprocessor(
    features: dict,
    binary_columns: list,
    strategy: str = "iterative",
    scale: bool = True,
):
    """This function removes the target variable from the feature list and creates a ColumnTransformer that imputes and encodes features:
    - Numeric: IterativeImputer (or SimpleImputer) + StandardScaler (z-score normalization)
    - Categorical: most_frequent imputer + OneHotEncoder(handle_unknown='ignore') only for multi-class categorical features, most_frequent imputer only for binary categorical features
    The transformer placed as the first step of a Pipeline passed to GridSearchCV (it will be fit per CV fold).
    Args:
        features (dict): dictionary with the features split in numerical and categorical
        strategy (str): imputation strategy for numerical features, options: 'iterative' or 'mean'
        scale (bool): whether to include StandardScaler in the numeric pipeline (default: True, False for tree-based models (RF))
    Returns:
        preprocessor (ColumnTransformer): sklearn ColumnTransformer with the preprocessing steps
    """
    num_cols = list(features.get("num", []))
    cat_cols = list(features.get("cat", []))
    bin_cols = binary_columns
    # Remove potential target column from categorical list
    target = features.get("target")
    if target in cat_cols:
        cat_cols.remove(target)

    # Separate binary and multi-class categorical columns
    # Define binary categorical columns, but keep only those that are actually in cat_cols
    #binary_cols_all = ["basis_sex", "a_ses_mig_status", "a_ses_partner", "cvd"]
    binary_cols = [c for c in bin_cols if c in cat_cols]
    multi_cols = [c for c in cat_cols if c not in binary_cols]

    # Numeric imputer + scaler
    num_imputer = IterativeImputer(random_state=42) if strategy == "iterative" else SimpleImputer(strategy="mean")

    numeric_steps = [("imputer", num_imputer)]
    if scale: # include StandardScaler only if scale=True for all models except tree-based ones (RF)
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)

    # Categorical imputer + one hot encoding for multi-class categorical
    categorical_pipeline_multi = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)), # try sparse=False if it does not work
        ]
    )
    categorical_pipeline_bin = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("multi_cat", categorical_pipeline_multi, multi_cols),
            ("bin_cat", categorical_pipeline_bin, binary_cols)
            ],
        remainder="drop",
    )
    return preprocessor


def perform_preprocessing_to_array(X, best_selected_pipe):
    """This function performs preprocessing on the dataset, builds an imputer on training set
    and infer imputation on validation and test set:
    Args:
        X (DataFrame): input dataframe to preprocess (options: X_train, X_val, X_test)
        best_selected_pipe: best selected pipeline from select_model function
    Returns:
        X_proc (NumPy array): preprocessed input features
    Note: Not used now because we don't need the output in this format.
    Since OHE creates new columns, we need to keep track of the original column names and their mappings,
    so we use the perform_preprocessing_to_df function instead to be given to SHAP.
    """
    pipe = best_selected_pipe["classifier"]  
    pre = pipe.named_steps["preprocessing"]
    X_proc = pre.transform(X)

    return X_proc


def perform_preprocessing(X, best_selected_pipe):
    """This function performs preprocessing on the dataset, builds an imputer on training set
    and infer imputation on validation and test set:
    Args:
        X (DataFrame): input dataframe to preprocess (options: X_train, X_val, X_test)
        best_selected_pipe: best selected pipeline from select_model function
    Returns:
        X_proc (DataFrame): preprocessed input features.
    Note: Since OHE creates new columns, we need to keep track of the original column names and their mappings.
    """

    pipe = best_selected_pipe["classifier"]
    pre = pipe.named_steps["preprocessing"]

    X_proc = pre.transform(X)
    feature_names = pre.get_feature_names_out()
    return pd.DataFrame(X_proc, columns=feature_names, index=X.index)
