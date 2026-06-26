from src.inference import infer_data
from src.evaluation_metric import compute_metrics
from src.evaluation_metric import extract_global_metrics
from typing import Tuple, Dict
import pandas as pd


def evaluate_model(
    classifiers: list,
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[Dict, Dict]:
    """This function eveluate the tuned model on a new dataset (validation or test)
    in order to do model comparison
    Args:
        classifiers (list): list of dictionaries with the tuned models selected with the GridSearchCV
        X (DataFrame): input features, options: X_val or X_test
        y (series): target variable, options: y_val or y_test
    Return:
        model_comparison_report (dict): dictionary with model name as key and dictionary with different metrics as values
        model_metrics_weighted_dict (dict): dictionary with only accuracy and weighted metrics for report
    """
    model_comparison_report = {}
    model_metrics_weighted_dict = {}
    for classifier in classifiers:
        model_name = list(classifier["classifier"].named_steps.keys())[-1]
        # inference
        y_pred = infer_data(
            best_estimator=classifier["classifier"],
            X_to_infer=X,
        )
        # compute metrics
        report = compute_metrics(
            y=y,
            y_pred=y_pred,
        )
        # get the full report for all metrics
        model_comparison_report[model_name] = report

         # extract global metrics (single source of truth)
        model_metrics_weighted_dict[model_name] = extract_global_metrics(report)

    return (
        model_comparison_report,
        model_metrics_weighted_dict,
    )
