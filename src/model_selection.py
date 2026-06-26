from sklearn.model_selection import GridSearchCV
from typing import Optional, Dict, Tuple, List, Any
import operator
import pandas as pd

import torch
import sys

sys.path.append("../src")
import src.model_class


def tune_and_select_model(
    X: pd.DataFrame,
    y: pd.Series,
    models: list,
) -> List[Dict]:
    """This function performs GridSearchCV on all the input model indipendentely. For each calss of models, it performs hyperparameter optimization on 5-fold cv.
    The goal if to perform a separate grid search on each class of model, and select the best tuned one for each class, and then use these for model selection on validation set.
    Args:
        X (DataFrame): input features
        y (Series): target variable
        models (list): list of models
    Return:
        results (List(Dict)): list of best tuned model for each class of models. The dictionary contains the name and parameters of the tuned models
    """

    results = []

    for m in models:
        # Perform grid search using cross-validation
        grid_search = GridSearchCV(
            estimator=m.pipeline,
            param_grid=m.params,
            cv=5,
            return_train_score=True,
            refit=True,
            verbose=1,
            n_jobs=-1,
        )
        # Adjust the CV fold as needed
        if isinstance(m, src.model_class.NeuralNetwork):
            print("Changing dtype of the target variable")
            y_tensor = torch.tensor(y.values, dtype=torch.long)
            grid_search.fit(X, y_tensor)
        else:
            grid_search.fit(X, y)

        results.append(
            {
                "classifier": grid_search.best_estimator_,
                "best score": grid_search.best_score_,
                "best_parameters": grid_search.best_params_,
                "mean_train_score": grid_search.cv_results_["mean_train_score"][grid_search.best_index_],
                "std_train_score": grid_search.cv_results_["std_train_score"][grid_search.best_index_],
                "mean_test_score": grid_search.cv_results_["mean_test_score"][grid_search.best_index_], # should be the same as best score
                "std_test_score": grid_search.cv_results_["std_test_score"][grid_search.best_index_],
            }
        )
    # Sorting result by best score
    results = sorted(
        results,
        key=operator.itemgetter("best score"),
        reverse=True,
    )

    return results


def select_model(
    classifiers: list,
    model_comparison_report: dict,
    metric: str = "accuracy",
    dataset_name: Optional[str] = None,
) -> Optional[Tuple[str, Dict, Dict,]]:
    """This function selects the best model accordingly to the accuracy (model selection should be done on the validation set).
    Args:
        classifiers (list): list of dictionaries with the tuned models selected with the GridSearchCV
        model_comparison_report (dict): dictionary with model name as key and dictionary with different metrics as values
        metric (str): name of the metric chosen for model selection
        dataset_name (str): name of the dataset, option: val or test
    Returns:
        best_model_name (str): best model name
        best_paramters (dict): best model parameters
        selected_model (dict): best model pipeline
    Note:
    Not used  in the current version, as we are using bootstrapping for model selection,
    but can be used as a backup or in case of small datasets where bootstrapping is not feasible.
    """
    # Initialize variables to track the best model and accuracy
    best_model_name = None
    best_accuracy = 0.0

    for model_name, model_metrics in model_comparison_report.items():
        if model_metrics[metric] > best_accuracy:
            best_accuracy = model_metrics[metric]
            best_model_name = model_name

    if dataset_name is not None:
        print(f"Working on {dataset_name} set")

    if best_model_name is None:
        print(f"No models with {metric} found in the model_comparison_report.")
        return None

    # Find the selected model
    selected_model = next(
        (
            classifier
            for classifier in classifiers
            if classifier["classifier"].named_steps.get(best_model_name)
        ),
        None,
    )
    if selected_model:
        print(
            f"The model with the highest {metric} is '{best_model_name}' with an {metric} of {best_accuracy:.4f}."
        )
        print("Selected model:", selected_model)
    else:
        print(f"No matching classifier found for the best model '{best_model_name}'.")

    best_paramters = selected_model["best_parameters"]
    return (
        best_model_name,
        best_paramters,
        selected_model,
    )

def select_model_from_bootstrapping(
    results_bootstrapping: dict = None,
    selection_metric: str = "accuracy",
    classifiers: list = None,
    selection_cutoff: float = 0.6,
):
    """Select the best model according to bootstrap P(best) for a given selection_metric.
    Args:
        - results_bootstrapping (dict): dictionary with the results of the bootstrapping procedure (joint_bootstrap_metrics function),
        - selection_metric (str): name of the metric chosen for model selection, e.g. "accuracy", "f1_weighted", etc.
        - classifiers (list): list of dictionaries with the tuned models selected with the GridSearchCV, used to match the selected model name to the actual pipeline for return
        - selection_cutoff (float): threshold for automatic selection of the best model based on P(best), default 0.6 (weak preference or better)
    Returns:
        - best_model_name (str): name of the model with highest P(best) for the selection metric
        - selected_model (object or None): matching classifier pipeline from `classifiers` or None
        - best_parameters (dict or None): best parameters of the selected model or None
        - p_value (float): P(best) for the chosen model
        - label (str): interpretation of p_value (indistinguishable / weak / moderate / ...)
    """

    # Basic validation
    if results_bootstrapping is None:
        raise ValueError("results_bootstrapping must be provided")
    if "p_best" not in results_bootstrapping:
        raise KeyError("'p_best' not found in results_bootstrapping")
    p_best = results_bootstrapping["p_best"]
    mean_rank = results_bootstrapping.get("mean_rank", None)

    if selection_metric not in p_best:
        raise KeyError(f"Metric '{selection_metric}' not present in p_best keys: {list(p_best.keys())}")

    # print overall info (optional)
    print(f"Bootstrap p-value for best model vs others (all metrics): {p_best}")
    if mean_rank is not None:
        print(f"Bootstrap mean rank for each model: {mean_rank}")

    # select best model for the chosen metric
    best_model_name = max(p_best[selection_metric], key=p_best[selection_metric].get)
    p_value = float(p_best[selection_metric][best_model_name])
    label = evidence_label(p_value)

    # print selection-metric-specific info
    print(f"\nMetric used for selection: '{selection_metric}'")
    print(f"Best model by P(best): {best_model_name} (P = {p_value:.4f})")
    print(f"Evidence level: {label}")

    # selection rule: always select if p > selection_cutoff (default 0.6)
    selected_model = None
    best_parameters = None
    if p_value > selection_cutoff:
        if not classifiers:
            print("Warning: classifiers list is empty or None; cannot match a pipeline to the selected model.")
        else:
            selected_model = next(
                (
                    classifier
                    for classifier in classifiers
                    if classifier.get("classifier") is not None
                    and classifier["classifier"].named_steps.get(best_model_name)
                ),
                None,
            )
            if selected_model is None:
                best_parameters = None
                print("Warning: selected model name found, but no matching classifier pipeline was found in `classifiers`.")
            else:
                print(f"Model '{best_model_name}' selected (p value > {selection_cutoff}).")
                best_parameters = selected_model["best_parameters"]
                
    else:
        print(f"Model not automatically selected because p value = {p_value:.4f} ≤ {selection_cutoff} (indistinguishable / weak evidence regime).")

    # return both the candidate name and the matched pipeline (or None), plus stats
    return best_model_name, selected_model, best_parameters, p_value, label

def evidence_label(p: float) -> str:
    """This function converts a p-value into a qualitative label of evidence strength, based on common thresholds.
    Args:
    - p (float): p-value to interpret
    Returns:
    - label (str): qualitative label of evidence strength
    """
    if p <= 0.50:
        return "indistinguishable"
    if p <= 0.60:
        return "borderline / indistinguishable"
    if p <= 0.70:
        return "weak evidence"
    if p <= 0.85:
        return "moderate evidence"
    if p <= 0.90:
        return "strong evidence"
    if p <= 0.95:
        return "very strong evidence"
    return "very strong / decisive evidence"

def check_selected_model(
    best_model_name: Optional[str],
    classifiers: Optional[List[Dict[str, Any]]],
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:

    if not classifiers:
        return best_model_name, None, None

    def _get_step_name(entry):
        pipe = entry.get("classifier")
        if pipe and hasattr(pipe, "named_steps"):
            return list(pipe.named_steps.keys())[-1]
        return None

    # Build simple lookup
    step_to_entry = {
        _get_step_name(entry): entry
        for entry in classifiers
        if _get_step_name(entry) is not None
    }

    # If manually provided name exists and matches
    if best_model_name in step_to_entry:
        selected_model = step_to_entry[best_model_name]
        return (
            best_model_name,
            selected_model,
            selected_model.get("best_parameters"),
        )

    # Fallback if None or not found
    fallback_order = ("XGBoost", "rfc")

    for fb in fallback_order:
        if fb in step_to_entry:
            selected_model = step_to_entry[fb]
            return (
                fb,
                selected_model,
                selected_model.get("best_parameters"),
            )

    # Nothing found
    return best_model_name, None, None
