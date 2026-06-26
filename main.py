from src.preprocessing import (
    delete_missing_values,
    perform_preprocessing,
)
from src.data_selection import (
    select_features,
    filter_feature,
)
from src.missing_type_utils import list_mcar_missing_type
from src.train_test_split import train_test_split_stratified
from src.model_class import create_model
from src.model_selection import (
    tune_and_select_model,
    select_model_from_bootstrapping,
    check_selected_model
)
from src.model_eveluation import evaluate_model
from src.utils import (
    init_parameters,
    get_json_params,
)
from src.bootstrapping import (
    joint_bootstrap_metrics,
    make_bootstrap_indices,
)
from src.model_explainability import (
    perform_permutation_feaure_importance,
    perform_shap,
)
from src.create_report_jsons import (
    generate_log,
    report_id_gen,
    create_jsons,
)
from reporting.mlee_report.generate_report import generate_report_with_jinja

import logging
import datetime
from auxiliary.turbopath import turbopath
import shutil
import copy


def main():
    # Generate a timestamp for the log file name
    startime = datetime.datetime.now()
    timestamp = report_id_gen(startime)
    output_dir = generate_log(timestamp)
    data, parameters = init_parameters()

    # Define target
    target = parameters["target"]

    # Calculate the number of samples for the subset
    subset_percentage = parameters["subset_percentage"]
    subset_size = int(len(data) * subset_percentage)
    if subset_percentage < 1.0:
        logging.info(
            "Subset created with %.2f %% of the original dataset",
            subset_percentage,
        )
    # Create a subset of the data
    data = data.sample(n=subset_size, random_state=42)

    # Feature Selection
    selected_data = select_features(
        data, get_json_params(parameters, "columns_to_keep")
    )

    # Filter data
    filtered_data = filter_feature(
        data=selected_data,
        dict_features=get_json_params(parameters, "filters"),
    )

    # Preprocessing
    # Missing values: Deletion of missing in the target variables and in the features used for stratification
    # list_mcar = list_mcar_missing_type(selected_data)
    # list_mcar = list(set(list_mcar + parameters["feature_stratification"]))
    # if target in list_mcar:
    #     list_mcar.remove(target)
    # else:
    #     print(f"{target} not found in the MCAR list.")

    processed_missing_data = delete_missing_values(
        data=filtered_data,
        target=target,
        list_columns=get_json_params(parameters, "feature_stratification"),
    )

    # Train_test split
    test_size = parameters["test_size"]
    validation_size = parameters["validation_size"]
    X_train, X_test, X_val, y_train, y_test, y_val = train_test_split_stratified(
        data=processed_missing_data,
        target=target,
        feature_stratification=get_json_params(parameters, "feature_stratification"),
        test_size=test_size,
        validation_size=validation_size,
    )
    logging.info(
        "X_train: %s, y_train: %s",
        str(X_train.shape),
        str(y_train.shape),
    )
    logging.info(
        "X_test: %s, y_test: %s",
        str(X_test.shape),
        str(y_test.shape),
    )
    logging.info(
        "X_val: %s, y_val: %s",
        str(X_val.shape),
        str(y_val.shape),
    )

    # Drop feature important for stratification but not for training
    X_train = X_train.drop(
        parameters["features_to_drop"],
        axis=1,
    )
    X_test = X_test.drop(
        parameters["features_to_drop"],
        axis=1,
    )
    X_val = X_val.drop(
        parameters["features_to_drop"],
        axis=1,
    )
    logging.info(
        "Columns useful for stratification, dropped during training: basis_uort, g_dgu_eu_m_u1",
    )
    logging.info(
        f"Number of feature used in traing: {len(X_train.columns)}",
    )
    logging.info(
        f"Set of feature used in traing: {X_train.columns}",
    )
    dict_features = copy.deepcopy(parameters["columns_to_keep"])

    to_remove = set(parameters["features_to_drop"]) | {parameters["target"]}

    for group in ["cat", "num"]:
        dict_features[group] = [
            c for c in dict_features[group]
            if c not in to_remove
        ]

    # Model definition
    models = create_model(
        model_definitions=parameters["models"],
        features=dict_features,
        binary_columns=parameters["binary_columns"],
        imputation_strategy=parameters["imputation_strategy"],
        y=y_train,  # only for neural network to set the output layer dimension 
    )
    logging.info("Models:")
    [logging.info(m.print()) for m in models]

    # Model tuning and first selection: select best hyperparamter-tuned model
    classifiers = tune_and_select_model(
        X=X_train,
        y=y_train,
        models=models,
    )
    # Model evaluation and selection on validation set
    # with bootstrapping for confidence intervals and ranking
    
    # Evaluation
    n_iterations = parameters["n_boot_iterations"]
    res_val = joint_bootstrap_metrics(
        classifiers=classifiers,
        X=X_val,
        y=y_val,
        output_dir=output_dir,
        metric_names=["accuracy", "precision", "recall", "f1_score"],
        n_iterations=n_iterations,
        seed=42,
        compute_ranking=True,
        make_indices_fn=make_bootstrap_indices, 
        ci=0.95,
    )
    ci_table_val = res_val["ci_table"]      # {metric: {model: {"mean", "ci"}}}

    # Bootstrap-based model selection (statistical decision)
    selection_metric = parameters["selection_metric"]
    selection_cutoff = parameters["selection_cutoff"]
    best_model_name, selected_model, best_parameters, p_value_boot, evidence_label = select_model_from_bootstrapping(
        results_bootstrapping=res_val,
        selection_metric=selection_metric,
        classifiers=classifiers,
        selection_cutoff=selection_cutoff,
    )
    # Fallback selection if bootstrap evidence is insufficient
    if selected_model is not None:
        selection_type = "automatic"
        selected_model_name = best_model_name
        print("Model selection successful.")
    else:
        print("Since model selection failed, fallback to manual model selection.")
        selection_type = "manual"
        selected_model_name, selected_model, best_parameters = check_selected_model(
            best_model_name=None,  # override to None since not automatically selected
            classifiers=classifiers,
        )
    print("Selected model name:", selected_model_name)
    print("Selected model:", selected_model) 
    print("Best parameters:", best_parameters)

    # Model evaluation on test set
    # with bootstrapping for confidence intervals
    res_test = joint_bootstrap_metrics(
    classifiers=classifiers,
    X=X_test,
    y=y_test,
    output_dir=output_dir,
    metric_names=["accuracy", "precision", "recall", "f1_score"],
    n_iterations=n_iterations,
    seed=42,
    compute_ranking=False,
    ci=0.95,
    )
    ci_table_test = res_test["ci_table"]

    # Model explainability
    # Feature Importance
    feature_importance_train_dict = perform_permutation_feaure_importance(
        classifiers=classifiers,
        X=X_train,
        y=y_train,
        dataset_name="train", 
        output_dir=output_dir,
    )

    feature_importance_test_dict = perform_permutation_feaure_importance(
        classifiers=classifiers,
        X=X_test,
        y=y_test,
        dataset_name="test",
        output_dir=output_dir,
    )

    # Extract preprocessed datasets for SHAP
    X_train_proc = perform_preprocessing(
        best_selected_pipe=selected_model,
        X=X_train,
    )   
    X_test_proc = perform_preprocessing(
        best_selected_pipe=selected_model,
        X=X_test,
    )

    # SHAP
    shap_output_prob = parameters["shap_output_prob"]
    perform_shap(
        models=models,
        classifiers=classifiers,
        best_model_name=selected_model_name,
        X=X_train_proc,
        X_before_preprocessing=X_train,
        y=y_train,
        shap_output_prob=shap_output_prob,
        dataset_name="train",
        output_dir=output_dir,
    )

    perform_shap(
        models=models,
        classifiers=classifiers,
        best_model_name=selected_model_name,
        X=X_test_proc,
        X_before_preprocessing=X_test,
        y=y_test,
        shap_output_prob=shap_output_prob,
        dataset_name="test",
        output_dir=output_dir,
    )

    # Get the execution time
    endtime = datetime.datetime.now()
    elapsed_time = endtime - startime
    logging.info(
        "Running Time: %s",
        elapsed_time,
    )

    # Create json file for reporting
    create_jsons(
        timestamp=timestamp,
        output_dir=output_dir,
        data=data,
        parameters=parameters,
        X_train=X_train,
        X_test=X_test,
        X_val=X_val,
        y_train=y_train,
        y_test=y_test,
        y_val=y_val,
        #imputation_strategy=parameters["imputation_strategy"],
        models=models,
        classifiers=classifiers,
        n_iterations=n_iterations,
        best_model_name=best_model_name,
        selection_metric=selection_metric,
        best_parameters=best_parameters,
        p_value_boot=p_value_boot,
        evidence_label=evidence_label,
        selection_type=selection_type,
        selected_model_name=selected_model_name,
        ci_table_val=ci_table_val,
        ci_table_test=ci_table_test,
        feature_importance_train_dict=feature_importance_train_dict,
        feature_importance_test_dict=feature_importance_test_dict,
        startime=startime,
        endtime=endtime,
        elapsed_time=elapsed_time,
    )

    # Creating the report
    identifier = str(timestamp) 
    generate_report_with_jinja(
        session_dir=output_dir,
        report_dir=turbopath("reports/" + identifier),
    )

    # creating reports zip file
    shutil.make_archive(
        base_name="reports/" + identifier,
        format="zip",
        root_dir="reports",
        base_dir=identifier,
    )

    print("Done!")


if __name__ == "__main__":
    main()
