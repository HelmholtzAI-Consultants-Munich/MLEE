from sklearn.pipeline import Pipeline
import pandas as pd


def infer_data(
    best_estimator: Pipeline,
    X_to_infer: pd.DataFrame,
) -> pd.Series:
    """This function perfomrs inference
    from the best estimator model obtained from the GridSearchCV
    Args:
        best_estimator (estimator object: Pipeline): best model and best hyperparamters set, output from the GridSearchCV
        X_to_infer (DataFrame): test set
    Returns:
    y_pre (array-like): The predicted values
    """

    y_pred = best_estimator.predict(X_to_infer)
    return y_pred
