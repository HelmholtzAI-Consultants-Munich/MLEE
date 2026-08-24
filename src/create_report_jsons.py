import logging
import os
import json
import datetime
import pandas as pd
from auxiliary.turbopath import turbopath
from codenamize import codenamize
from numpy.random import randint

from src.model_selection import evidence_label


def write_json(
    json_dict: dict,
    output_file: str,
):
    output_file = turbopath(output_file)
    os.makedirs(output_file.parent, exist_ok=True)

    # Serializing json
    json_object = json.dumps(json_dict, indent=4)

    # Writing to sample.json
    with open(output_file, "w") as outfile:
        outfile.write(json_object)


def setup_logging(log_file_name):
    # Configure logging with the specified log file
    logging.basicConfig(
        filename=log_file_name,
        encoding="utf-8",
        format="%(levelname)s:%(message)s",
        level=logging.DEBUG,
    )
    # Set the logging level for Matplotlib to WARNING to avoind DEBUG message due to font style
    matplotlib_logger = logging.getLogger("matplotlib")
    matplotlib_logger.setLevel(logging.WARNING)


def generate_log(
    timestamp: str,
) -> str:
    output_dir = turbopath("outputs/" + str(timestamp))
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir + "/meta", exist_ok=True)
    log_file_name = output_dir + "/meta/" + str(timestamp) + ".log"
    setup_logging(log_file_name)

    return output_dir


def report_id_gen(startime):
    timestamp = startime.strftime("%Y-%m-%d_%H%M%S_%f")[:-3]
    return timestamp


def create_jsons(
    timestamp: str,
    output_dir: str,
    data: pd.DataFrame,
    parameters: dict,
    n_instances_before: int,
    n_instances_after: int,
    percentage_retained: float,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    y_val: pd.Series,
    #imputation_strategy: str,
    models: list,
    classifiers: list,
    n_iterations: int,
    best_model_name: str,
    selection_metric: str,
    best_parameters: dict,
    p_value_boot: float ,
    evidence_label: str,
    selection_type: str,
    selected_model_name,
    ci_table_val: dict,
    ci_table_test: dict,
    feature_importance_train_dict: dict,
    feature_importance_test_dict: dict,
    startime: datetime.datetime,
    endtime: datetime.datetime,
    elapsed_time: datetime.datetime,
) -> None:
    """This function creates all the json files for the output and consequently for the report"""
    meta = {
        "timestamp": str(timestamp),
    }

    write_json(
        json_dict=meta,
        output_file=output_dir + "/meta/meta.json",
    )

    reported_parameters = {
        "parameters": parameters,
    }

    write_json(
        json_dict=reported_parameters,
        output_file=output_dir + "/meta/parameters.json",
    )

    columns_to_keep = parameters["columns_to_keep"]
    write_json(
        json_dict=columns_to_keep,
        output_file=output_dir + "/tables/columns_to_keep.json",
    )

    dataset = {
        "dimension": data.shape,
        "subset_percentage": parameters["subset_percentage"],
        "n_instances_before_filtering": n_instances_before,
        "n_instances_after_filtering": n_instances_after,
        "percentage_retained": percentage_retained,
    }

    write_json(
        json_dict=dataset,
        output_file=output_dir + "/dataset/dataset.json",
    )

    trainTest = {
        "Training": X_train.shape,
        "Validation": X_val.shape,
        "Test": X_test.shape,
    }

    write_json(
        json_dict=trainTest,
        output_file=output_dir + "/traintest/traintest.json",
    )

    # imputation_strategy = parameters["imputation_strategy"]
    # write_json(
    #     json_dict=imputation_strategy,
    #     output_file=output_dir + "/imputation/imputation.json",
    # )

    model = [m.print() for m in models]
    write_json(
        json_dict=model,
        output_file=output_dir + "/modelDefinition/models.json",
    )

    classifiersScores = {}
    for classifier in classifiers:
        model_name = list(classifier["classifier"].named_steps.keys())[-1]
        classifiersScores[model_name] = {}
        for n in [
            "best_parameters",
            "mean_train_score",
            "std_train_score",
            "mean_test_score",
            "std_test_score",
        ]:
            classifiersScores[model_name][n] = classifier[n]

    write_json(
        json_dict=classifiersScores,
        output_file=output_dir + "/modelTuning/classifiers_scores.json",
    )

    bestClassifier = {
        "best_model_name": best_model_name,
        "best_tuned_parameters": best_parameters,
        "selection_metric": selection_metric,
        "p_value_boot": p_value_boot,
        "evidence_label": evidence_label,
        "selection_type": selection_type,
        "selected_model_name": selected_model_name,
    }

    write_json(
        json_dict=bestClassifier,
        output_file=output_dir + "/modelSelection/bestclassifier.json",
    )

    write_json(
        json_dict=n_iterations,
        output_file=output_dir + "/bootstrapping/n_iterations.json",
    )

    write_json(
        json_dict=ci_table_val,
        output_file=output_dir + "/bootstrapping/ci_table_val.json",
    )

    write_json(
        json_dict=ci_table_test,
        output_file=output_dir + "/bootstrapping/ci_table_test.json",
    )

    for model_name, importance_values in feature_importance_train_dict.items():
        # Support both old format (dict of columns -> {idx: val}) and new tidy format
        if isinstance(importance_values, dict) and "features" in importance_values:
            feature_importance_train = {
                "features": list(importance_values.get("features", [])),
                "importances": list(importance_values.get("importances", [])),
                "std": list(importance_values.get("std", [])),
            }
        else:
            # old format: {'Feature': {idx: name}, 'Mean Importance': {idx: val}, ...}
            feature_importance_train = {
                "features": list(importance_values.get("Feature", {}).values()),
                "importances": list(importance_values.get("Mean Importance", {}).values()),
                "std": list(importance_values.get("Standard Deviation Importance", {}).values()),
            }
        write_json(
            json_dict=feature_importance_train,
            output_file=output_dir
            + "/tables/importance/"
            + model_name
            + "_feature_importance_train.json",
        )

    for model_name, importance_values_test in feature_importance_test_dict.items():
        if isinstance(importance_values_test, dict) and "features" in importance_values_test:
            feature_importance_test = {
                "features": list(importance_values_test.get("features", [])),
                "importances": list(importance_values_test.get("importances", [])),
                "std": list(importance_values_test.get("std", [])),
            }
        else:
            feature_importance_test = {
                "features": list(importance_values_test.get("Feature", {}).values()),
                "importances": list(importance_values_test.get("Mean Importance", {}).values()),
                "std": list(importance_values_test.get("Standard Deviation Importance", {}).values()),
            }

        write_json(
            json_dict=feature_importance_test,
            output_file=output_dir
            + "/tables/importance/"
            + model_name
            + "_feature_importance_test.json",
        )

    reported_time = {
        "start_time": startime.strftime("%Y-%m-%d_%H%M%S"),
        "end_time": endtime.strftime("%Y-%m-%d_%H%M%S"),
        "elapsed_time": str(elapsed_time),
    }

    write_json(
        json_dict=reported_time,
        output_file=output_dir + "/meta/times.json",
    )
