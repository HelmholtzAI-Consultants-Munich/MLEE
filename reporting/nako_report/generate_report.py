from jinja2 import Environment, FileSystemLoader, select_autoescape

from reporting.mlee_report.staging import create_stage, move_from_stage
from auxiliary.turbopath import turbopath

import json
import os


# import subprocess

def custom_zip(list1, list2):
    return zip(list1, list2)

def generate_report_with_jinja(
    session_dir: str,
    report_dir: str = "reports",
    stage_dir: str = "reporting/the_stage",
):
    # stage
    session_dir = turbopath(session_dir)
    stage_dir = turbopath(stage_dir)
    create_stage(
        session_dir=session_dir,
        stage_dir=stage_dir,
    )

    # template
    env = Environment(
        loader=FileSystemLoader(stage_dir),
        autoescape=select_autoescape(["html", "xml"]),
    )
    env.globals.update(custom_zip=custom_zip)

    # meta
    with open(stage_dir + "/session/meta/meta.json") as f:
        meta = json.load(f)

    # parameters
    with open(stage_dir + "/session/meta/parameters.json") as f:
        parameters = json.load(f)

    with open(stage_dir + "/session/tables/columns_to_keep.json") as f:
        columns_to_keep = json.load(f)

    # time
    with open(stage_dir + "/session/meta/times.json") as f:
        times_log = json.load(f)
    print(times_log)

    # dataset
    with open(stage_dir + "/session/dataset/dataset.json") as f:
        dataset = json.load(f)

    # train-test
    with open(stage_dir + "/session/traintest/traintest.json") as f:
        traintest = json.load(f)

    # # imputation (miceforest - not applicable in this version)
    # # method
    # with open(stage_dir + "/session/imputation/imputation.json") as f:
    #     imputation = json.load(f)
    # # plots
    # imputation_folder = stage_dir + "/session/imputation/"
    # train_imputation_exists = os.path.isfile(os.path.join(imputation_folder, 'train_imputation.png'))
    # test_imputation_exists = os.path.isfile(os.path.join(imputation_folder, 'test_imputation.png'))

    # model-definition
    with open(stage_dir + "/session/modelDefinition/models.json") as f:
        models = json.load(f)

    # model-tuning
    with open(stage_dir + "/session/modelTuning/classifiers_scores.json") as f:
        classifiers_scores = json.load(f)

    # model-selection
    with open(stage_dir + "/session/modelSelection/bestclassifier.json") as f:
        bestclassifier = json.load(f)

    # model evaluation - bootstrapping
    with open(stage_dir + "/session/bootstrapping/n_iterations.json") as f:
        n_iterations = json.load(f)

    with open(stage_dir + "/session/bootstrapping/ci_table_val.json") as f:
        ci_table_val = json.load(f)

    with open(stage_dir + "/session/bootstrapping/ci_table_test.json") as f:
        ci_table_test = json.load(f)
    
    plot_boot_folder_path = stage_dir + "/session/bootstrapping/"
    all_files_bootstrapping = [
    f for f in os.listdir(plot_boot_folder_path)
    if f.endswith(".png")
]

    # model-explainability-train
    # table
    feature_importance_data_train = {}
    tables_folder_path = stage_dir + "/session/tables/importance/"
    all_tables = os.listdir(tables_folder_path)
    all_tables_train = [file for file in all_tables if "train" in file]
    model_names = [file.split("_")[0] for file in all_tables_train]
    for model_name in model_names:
        with open(stage_dir + "/session/tables/importance/" + model_name + "_feature_importance_train.json") as f:
            feature_importance_train = json.load(f)
        feature_importance_data_train[model_name] = feature_importance_train
    # plot
    all_files = []
    for model_name in model_names:
        plot_folder_path = stage_dir + "/session/importance/" + model_name + "/"
        files = os.listdir(plot_folder_path)
        all_files.extend(files)
    all_files_train = [file for file in all_files if "train" in file]


    # model-explainability-test
    # table
    feature_importance_data_test= {}
    all_tables_test = [file for file in all_tables if "test" in file]
    model_names = [file.split("_")[0] for file in all_tables_test]
    for model_name in model_names:
        with open(stage_dir + "/session/tables/importance/" + model_name + "_feature_importance_test.json") as f:
            feature_importance_test = json.load(f)
        feature_importance_data_test[model_name] = feature_importance_test 
    # plot
    all_files_test = [file for file in all_files if "test" in file]


    # main
    templateMain = env.get_template("blueprint_main.html")
    outputHtml = templateMain.render(
        meta=meta,
        parameters=parameters,
        columns_to_keep=columns_to_keep,
        dataset=dataset,
        traintest=traintest,
        # imputation=imputation,
        # train_imputation_exists=train_imputation_exists,
        # test_imputation_exists=test_imputation_exists,
        models=models,
        classifiers_scores=classifiers_scores,
        n_iterations=n_iterations,
        ci_table_val=ci_table_val,
        bestclassifier=bestclassifier,
        ci_table_test=ci_table_test,
        all_files_bootstrapping=all_files_bootstrapping,
        feature_importance_data_train=feature_importance_data_train,
        all_files_train=all_files_train,
        feature_importance_data_test=feature_importance_data_test,
        all_files_test=all_files_test,
        times=times_log,
    )

    # write html
    outputHtmlPath = stage_dir + "/mlee.html"
    with open(outputHtmlPath, "w") as fm:
        fm.write(outputHtml)

    move_from_stage(
        stage_dir=stage_dir,
        output_dir=report_dir,
    )


if __name__ == "__main__":
    generate_report_with_jinja(session_dir="input/test_report_session")
