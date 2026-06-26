import json
import logging
import pandas as pd
from typing import Tuple, Dict

def flatten(l):
    """This function takes a list of lists and flattens it into a single list.
    Args:
    l (list): A list containing sublists.

    Returns:
    list: A new list containing all the elements from the sublists.
    """
    return [item for sublist in l for item in sublist]


def load_data(
    file_path: str,
) -> None:
    """
    This function loads data from a text file into a DataFrame
    Args:
        file_path (str): path to the text file
    Returns:
        data (DataFrame): loaded data as a DataFrame
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None


def merge_parameters(
    default_params: dict,
    user_params: dict,
) -> dict:
    """
    This function merges two dictionaries while keeping parameters from the JSON file and using default values for missing keys.
    Args:
        default params (dict): default parameters defined in main.py and explained in the README.md
        user_params (dict): parameters defined by the user in the input_paramteres.json
    Returns:
        merged_params (dict): merged dictionary in which default parameters are used unless the user's parameters are updated.
    Notes:
        If a key is present in both dictionaries, the value from user_params will overwrite the value from default_params.
    """
    # Check keys provided by user
    default_params_keys = list(default_params.keys())
    for key in list(user_params.keys()):
        if key not in default_params_keys:
            logging.warning(f"Key {key} not valid")
    # Create a copy of the default parameters dictionary
    merged_params = default_params.copy()
    # Update the copy with the user parameters, overwriting any common keys
    if user_params is not None:
        merged_params.update(user_params)
    return merged_params


def init_parameters() -> (
    Tuple[
        pd.DataFrame,
        Dict,
    ]
):
    # Read parameters
    with open("input_parameters.json") as f:
        user_parameters = json.load(f)
    with open("default_parameters.json") as f2:
        default_parameters = json.load(f2)

    # Merge default and users parameters
    parameters = merge_parameters(default_parameters, user_parameters)
    logging.info("Starting parameters: %s", str(parameters))

    # Load the data
    try:
        data = load_data(parameters["path_name"] + parameters["file_name"])
        logging.info("Dimension of the original dataset: %s", str(data.shape))
    except Exception as e:
        logging.error("An error occurred while loading the data: %s", str(e))
    return data, parameters


def get_json_params(params, key):
    if key in params:
        return params[key]
    else:
        return None