from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as sklearn_LR
from sklearn.linear_model import ElasticNet as sklearn_ElasticNet
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from src.preprocessing import make_preprocessor
from xgboost import XGBClassifier
from src.utils import merge_parameters

import numpy as np
import shap
import torch
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)

class Model:
    def __init__(self) -> None:
        self.random_state = 42
        self.pipeline = None
        self.params = {}

    def create_pipeline(self, features, binary_columns, imputation_strategy):
        raise NotImplementedError("This method should be implemented in the child class")

    def get_explainer(self, classifier_instance):
        return None

    def print(self):
        print("No specific info")


class RandomForest(Model):
    default_params = {
        "n_estimators": [100, 200, 500, 1000],
        "max_depth": [2, 5, 10,],  # careful with setting max_depth to None, caus it will likely lead to overfitting
        "max_features": [2, "sqrt", "log2"],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True],
    }

    def __init__(self, parameters) -> None:
        super().__init__()
        self.merged_params = merge_parameters(RandomForest.default_params, parameters)

    def create_pipeline(self, features, binary_columns, imputation_strategy):
        preprocessor = make_preprocessor(features=features, binary_columns=binary_columns, strategy=imputation_strategy, scale=False)
        self.pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("rfc", RandomForestClassifier())
        ])
        self.params = {}
        for key, val in self.merged_params.items():
            self.params["rfc__" + key] = val

    def get_explainer(self, classifier_instance):
        return shap.TreeExplainer(classifier_instance)

    def print(self):
        string_to_print = str(f"------- RandomForest -------")
        for key, val in self.merged_params.items():
            string_to_print += f"\n {key} = {val}"
        print(string_to_print)
        return string_to_print


class LogisticRegression(Model):
    default_params = {
        "penalty": ["l1", "l2", "elasticnet", None],
        "C": [0.1, 1.0, 10.0],
        "solver": ["saga"],  # other options are not available for now because of incompatibility with penalty
        "max_iter": [100, 200, 500],
        "l1_ratio": [0.2, 0.5, 0.7],
    }

    def __init__(self, parameters) -> None:
        super().__init__()
        self.merged_params = merge_parameters(
            LogisticRegression.default_params, parameters
        )

    def create_pipeline(self, features, binary_columns, imputation_strategy):
        preprocessor = make_preprocessor(features=features, binary_columns=binary_columns, strategy=imputation_strategy, scale=True)
        self.pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("LR", sklearn_LR()),
        ])
        self.params = {}
        for key, val in self.merged_params.items():
            self.params["LR__" + key] = val

    def print(self):
        string_to_print = str(f"------- LogisticRegression -------")
        for key, val in self.merged_params.items():
            string_to_print += f"\n {key} = {val}"
        print(string_to_print)
        return string_to_print


class SVMClassifier(Model):
    default_params = {
        "kernel": ["linear", "rbf", "sigmoid"],
        "C": [0.1, 1.0, 10.0],
        "gamma": ["scale", "auto", 0.1, 1.0],
    }

    def __init__(self, parameters) -> None:
        super().__init__()
        self.merged_params = merge_parameters(SVMClassifier.default_params, parameters)

    def create_pipeline(self, features, binary_columns, imputation_strategy):
        preprocessor = make_preprocessor(features=features, binary_columns=binary_columns, strategy=imputation_strategy, scale=True)
        self.pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("SVM", SVC())
        ])
        self.params = {}
        for key, val in self.merged_params.items():
            self.params["SVM__" + key] = val

    def print(self):
        string_to_print = str(f"------- SVMClassifier -------")
        for key, val in self.merged_params.items():
            string_to_print += f"\n {key} = {val}"
        print(string_to_print)
        return string_to_print


class ElasticNet(Model):  # for regression
    default_params = {
        "alpha": [0.1, 1.0, 10.0],
        "l1_ratio": [0.2, 0.5, 0.8],
        "max_iter": [100, 200, 500],
    }

    def __init__(self, parameters) -> None:
        super().__init__()
        self.merged_params = merge_parameters(ElasticNet.default_params, parameters)

    def create_pipeline(self, features, binary_columns, imputation_strategy):
        preprocessor = make_preprocessor(features=features, binary_columns=binary_columns, strategy=imputation_strategy, scale=True)
        self.pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("ElasticNet", sklearn_ElasticNet()),
        ])
        self.params = {}
        for key, val in self.merged_params.items():
            self.params["ElasticNet__" + key] = val

    def print(self):
        string_to_print = str(f"------- ElasticNet -------")
        for key, val in self.merged_params.items():
            string_to_print += f"\n {key} = {val}"
        print(string_to_print)
        return string_to_print


class KNN(Model):
    default_params = {
        "n_neighbors": [3, 5, 7, 10],
        "metric": ["manhattan"]}

    def __init__(self, parameters) -> None:
        super().__init__()
        self.merged_params = merge_parameters(KNN.default_params, parameters)

    def create_pipeline(self, features, binary_columns, imputation_strategy):
        preprocessor = make_preprocessor(features=features, binary_columns=binary_columns, strategy=imputation_strategy, scale=True)
        self.pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("KNN", KNeighborsClassifier()),
        ])
        self.params = {}
        for key, val in self.merged_params.items():
            self.params["KNN__" + key] = val

    # def get_explainer(self, classifier_instance, data=None):
    #     return shap.KernelExplainer(classifier_instance.predict, data)

    def print(self):
        string_to_print = str(f"------- KNN -------")
        for key, val in self.merged_params.items():
            string_to_print += f"\n {key} = {val}"
        print(string_to_print)
        return string_to_print


class XGBoost(Model):
    default_params = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.1, 0.01, 0.001],
        "n_estimators": [100, 200, 500],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "gamma": [0, 1, 5],
        "n_jobs": [1],
        # "tree_method": ["exact"] ### -> check this
    }

    def __init__(self, parameters) -> None:
        super().__init__()
        self.merged_params = merge_parameters(XGBoost.default_params, parameters)

    def create_pipeline(self, features, binary_columns, imputation_strategy):
        preprocessor = make_preprocessor(features=features, binary_columns=binary_columns, strategy=imputation_strategy, scale=True)
        self.pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("XGBoost", XGBClassifier())
        ])
        self.params = {}
        for key, val in self.merged_params.items():
            self.params["XGBoost__" + key] = val

    def print(self):
        string_to_print = str(f"------- XGBoost -------")
        for key, val in self.merged_params.items():
            string_to_print += f"\n {key} = {val}"
        print(string_to_print)
        return string_to_print


class ClassifierModule(nn.Module):
    def __init__(self, num_layers: int, num_units: int, dropout: float, num_classes: int):
        super().__init__()

        self.num_layers = max(1, num_layers)
        self.input_layer = nn.LazyLinear(num_units)
        self.dropout = nn.Dropout(dropout)

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(num_units, num_units) for _ in range(self.num_layers - 1)]
        )
        self.output = nn.Linear(num_units, num_classes)

    def forward(self, X, **kwargs):
        # Convert input data to torch.float32
        X = X.float()
        X = F.relu(self.input_layer(X))
        X = self.dropout(X)

        for layer in self.hidden_layers:
            X = F.relu(layer(X))
            X = self.dropout(X)

        return self.output(X) # raw logits, not softmax probabilities


class NeuralNetwork(Model):
    default_params = {
        "module__num_layers": [1, 2, 5, 10],
        "module__num_units": [10, 20],
        "module__dropout": [0.0, 0.5],
        "optimizer__lr": [0.05, 0.1],
        "optimizer__nesterov": [True, False],
    }

    def __init__(self, parameters) -> None:
        super().__init__()
        self.merged_params = merge_parameters(NeuralNetwork.default_params, parameters)

    def create_pipeline(self, features, binary_columns, imputation_strategy, num_classes):
        preprocessor = make_preprocessor(features=features, binary_columns=binary_columns, strategy=imputation_strategy, scale=True)
        
        self.pipeline = Pipeline([
            ("preprocessing", preprocessor),
            (
                "Net",
                NeuralNetClassifier(
                    module=ClassifierModule,
                    module__num_classes=num_classes,
                    max_epochs=20,
                    callbacks=[EarlyStopping(patience=5)],
                    optimizer=torch.optim.SGD,
                    criterion=torch.nn.CrossEntropyLoss,
                    optimizer__momentum=0.9,
                    train_split=ValidSplit(0.2),
                    verbose=0
                ),
            )
            ]
        )
        self.params = {}
        for key, val in self.merged_params.items():
            self.params["Net__" + key] = val

    def print(self):
        string_to_print = str(f"------- NeuralNetwork -------")
        for key, val in self.merged_params.items():
            string_to_print += f"\n {key} = {val}"
        print(string_to_print)
        return string_to_print


# Factory
def create_model(
    model_definitions: dict,
    features: dict,
    binary_columns: list,
    imputation_strategy: str,
    y: list,
) -> list:
    models = []
    for m, p in model_definitions.items():
        if "LogisticRegression" == m:
            models.append(LogisticRegression(p))
        elif "SVMClassifier" == m:
            models.append(SVMClassifier(p))
        elif "RandomForest" == m:
            models.append(RandomForest(p))
        elif "KNN" == m:
            models.append(KNN(p))
        elif "XGBoost" == m:
            models.append(XGBoost(p))
        elif "ElasticNet" == m:
            models.append(ElasticNet(p))
        elif "NeuralNetwork" == m:
            models.append(NeuralNetwork(p))

    num_classes = len(np.unique(y))

    for model in models:
        if isinstance(model, NeuralNetwork):
            model.create_pipeline(features, binary_columns, imputation_strategy, num_classes)
        else:
            model.create_pipeline(features, binary_columns, imputation_strategy)

    return models
