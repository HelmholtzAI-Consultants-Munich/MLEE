from sklearn.metrics import classification_report
import pandas as pd


def compute_metrics(
    y: pd.Series,
    y_pred: pd.Series,
) -> dict:
    """This function evaluates the model based on accuracy metric.
    Args:
        y (array-like): target varaible, ground truth, options:  y_val or y_test
        y_pred (aray-like): prediction resulted from the inference, options: y_val_pred or y_test_pred
    Returns:
        report (dict): dictionary with different metrics
    """
    report = classification_report(
        y_true=y,
        y_pred=y_pred,
        output_dict=True,
    )

    return report


def extract_global_metrics(
        report: dict
        ) -> dict:
    """
    Keep only global scalar metrics from a sklearn classification_report(output_dict=True)
    Args:
        report (dict): output of classification_report with output_dict=True
    Returns:
        reduced_report (dict): dictionary with only accuracy and weighted metrics for report
    """
    reduced_report = {}

    # accuracy is already scalar
    reduced_report["accuracy"] = report["accuracy"]

    # weighted average metrics
    wa = report.get("weighted avg", {})
    reduced_report["precision"] = wa.get("precision")
    reduced_report["recall"] = wa.get("recall")
    reduced_report["f1_score"] = wa.get("f1-score")

    return reduced_report