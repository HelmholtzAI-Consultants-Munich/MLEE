import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns
from pathlib import Path

from sklearn.utils import resample
from src.evaluation_metric import compute_metrics, extract_global_metrics
from typing import List, Dict, Tuple

# Optional parallel tools
# TO DO: can be used to speed up the bootstrapping by parallelizing the iterations,
# but need to be tested and integrated carefully to avoid issues with random state and reproducibility.
try:
    from joblib import Parallel, delayed
except Exception:
    Parallel = None
    delayed = None


# ---- Relative model comparison ----

def joint_bootstrap_metrics(
    classifiers: List[Dict],
    X,
    y,
    output_dir: str,
    metric_names: List[str] = ["accuracy", "precision", "recall", "f1_score"],
    n_iterations: int = 2000,
    seed: int = 42,
    compute_ranking: bool = True,
    make_indices_fn = None,
    ci: float = 0.95,
):
    """
    This function performs a single-pass joint bootstrap that returns per-model metric samples (for CI/table)
    and optional ranking summaries (p_best, mean_rank, rank_matrices) necessary for model evaluation on the validaiton set.
    Returns a dict with:
      - metrics_by_model: { metric_name: { model_name: [val_b1, ...] } }
      - ci_table: { metric_name: { model_name: {"mean":..., "ci": (lo,hi)} } }
      - rank_matrices: { metric_name: np.ndarray or None }
      - p_best: { metric_name: { model_name: p_best } } or None
      - mean_rank: { metric_name: { model_name: mean_rank } } or None
      - model_names, boot_indices
    """

    # prepare indices generator
    N = len(y)
    if make_indices_fn is None:
        make_indices_fn = make_bootstrap_indices
    boot_indices = make_indices_fn(N, n_iterations, seed)

    # precompute model names and predictions once
    model_names = []
    preds = []
    for clf in classifiers:
        model = clf["classifier"]
        name = list(model.named_steps.keys())[-1] if hasattr(model, "named_steps") else str(model)
        model_names.append(name)
        y_pred_full = model.predict(X)
        if hasattr(y_pred_full, "to_numpy"):
            y_pred_full = y_pred_full.to_numpy()
        preds.append(y_pred_full)

    n_models = len(preds)

    # containers: metrics_by_model[metric][model] -> list
    metrics_by_model: Dict[str, Dict[str, List[float]]] = {
        m: {name: [] for name in model_names} for m in metric_names
    }

    # rank matrices per metric (only if compute_ranking)
    rank_matrices = {m: (np.empty((n_iterations, n_models), dtype=int) if compute_ranking else None)
                     for m in metric_names}

    # single loop over bootstrap replicates
    for b, inds in enumerate(boot_indices):
        # sample y once
        y_b = y.iloc[inds] if hasattr(y, "iloc") else y[inds]

        # compute reduced report per model
        reduced_per_model = []
        for m_idx in range(n_models):
            ypred_b = preds[m_idx][inds]
            report = compute_metrics(y_b, ypred_b)
            reduced = extract_global_metrics(report)   # must contain keys for metric_names
            reduced_per_model.append(reduced)

        # for each metric_name, build metric vector and store
        for metric in metric_names:
            vals = np.array([ (reduced_per_model[i].get(metric) if reduced_per_model[i] is not None else np.nan)
                              for i in range(n_models)], dtype=float)
            # append to per-model lists
            for i, name in enumerate(model_names):
                metrics_by_model[metric][name].append(vals[i])

            # ranking for this metric
            if compute_ranking:
                order_desc = np.argsort(-vals, kind="stable")
                ranks = np.empty_like(order_desc)
                ranks[order_desc] = np.arange(1, n_models + 1)
                rank_matrices[metric][b, :] = ranks

    # compute CI tables using the helpers
    ci_table = {}
    for metric in metric_names:
        ci_table[metric] = summarize_metric_list(metrics_by_model[metric], ci=ci)
    
    ci_table_model_first = ci_table_to_model_first(ci_table)  

    # ranking summaries
    p_best = None
    mean_rank = None
    if compute_ranking:
        p_best = {}
        mean_rank = {}
        for metric in metric_names:
            p_best[metric] = {}
            mean_rank[metric] = {}
            rm = rank_matrices[metric]
            for i, name in enumerate(model_names):
                p_best[metric][name] = float((rm[:, i] == 1).mean())
                mean_rank[metric][name] = float(rm[:, i].mean())

    # plot bootstrap distribution for all metrics using plot_bootstrap_metrics
    for metric_name in metric_names:
        plot_bootstrap_metrics(
            metrics_by_model=metrics_by_model[metric_name],
            title=rf"Distribution of Bootstrapped $\mathbf{{{metric_name}}}$ by Model",
            output_file=output_dir + "/bootstrapping/" + metric_name +"_bootstrap_val.png",
        )
    return {
        "metrics_by_model": metrics_by_model,
        "ci_table": ci_table_model_first,  # model-first format for easier access in reports
        "rank_matrices": rank_matrices,
        "p_best": p_best,
        "mean_rank": mean_rank,
        "model_names": model_names,
        "boot_indices": boot_indices,
    }

def make_bootstrap_indices(
        N: int,
        n_iterations: int,
        seed: int = 42):
    """This function generates bootstrap indices for relative model comparison,
    ensuring that the same indices are used for all models in each iteration.
    Args:
        N (int): Number of samples in the dataset
        n_iterations (int): Number of bootstrap iterations
        seed (int): Random seed for reproducibility
    Returns:
        boot_indices (list): List of bootstrap indices for each iteration
    """
    rng = np.random.default_rng(seed)
    return [rng.integers(0, N, size=N) for _ in range(n_iterations)]


def summarize_metric_list(metrics_by_model_for_metric: Dict[str, List[float]], ci: float = 0.95) -> Dict[str, Dict]:
    """ Summarize bootstrap samples for a single metric across models.
    Args:
        metrics_by_model_for_metric: { model_name: [val_b1, val_b2, ...], ... }
        ci: confidence level
    Returns:
        summary: { model_name: {"mean": mean, "ci": (lo, hi) }, ... }
    """
    alpha = 1 - ci
    summary = {}
    for name, vals in metrics_by_model_for_metric.items():
        arr = np.array(vals, dtype=float)
        lo, hi = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        summary[name] = {"mean": float(arr.mean()), "ci": (float(lo), float(hi))}
    return summary

def plot_bootstrap_metrics(
        metrics_by_model: Dict[str, List[float]],
        title: str = None,
        output_file: str = None,
        ):
    """
    This function draws boxplots + jittered points to visualize
    the distribution of bootstrap metrics for each model
    and compare them visually.
    Args:
        metrics_by_model: {model_name: [val_b1, val_b2, ...]}
        title: Optional title for the plot
    Returns:
        None (displays the plot)
    """
    df = []
    for name, vals in metrics_by_model.items():
        for v in vals:
            df.append({"model": name, "metric": v})
    df = pd.DataFrame(df)

    plt.figure(figsize=(10, 5))
    sns.boxplot(x="model",
                y="metric",
                data=df,
                showcaps=True,
                boxprops={'facecolor':'none'},
                showfliers=False,
                )
    palette = sns.color_palette("flare_r", n_colors=df["model"].nunique())
    sns.stripplot(x="model",
                  y="metric",
                  hue="model",
                  data=df,
                  jitter=0.25,
                  alpha=0.6,
                  size=4,
                  palette=palette,
                  )
    plt.ylabel("Metric value")
    plt.xlabel("Model")
    if title:
        plt.title(title)
    plt.tight_layout()
    if output_file:
        try:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        plt.savefig(output_file)
    else:
        plt.show()

def ci_table_to_model_first(ci_table: dict) -> dict:
    """
    This function converts the ci_table from metric-first to model-first format
    for easier access when generating reports:
    From ci_table dict (metric -> model -> summary) to
    model-first dict (model -> metric -> summary).
    """
    model_first = {}
    for metric, models_dict in ci_table.items():
        for model_name, summary in models_dict.items():
            model_first.setdefault(model_name, {})[metric] = summary
    return model_first

# ---- Absolute performance ----

def summarize_bootstrap_list(
        metrics_list,
        ci=0.95) -> dict:
    """This function summarizes the bootstrapping results as a list of metrics and CIs
    Args:
        metrics_list (list): Results from bootstrapping_test function
        ci (float): Confidence interval for the bootstrap estimates
    Returns:
        out (dict): Summary of bootstrap results
    Note: Not used in the current version, as we are using the joint_bootstrap_metrics for relative model comparison,
    but can be used for absolute performance estimation in the bootstrapping_all_models function.
    """
    summary = {}
    alpha = 1 - ci
    wanted = ["accuracy", "precision", "recall", "f1_score"]
    for key in wanted:
        values = np.array([m[key] for m in metrics_list], dtype=float)
        lo, hi = np.percentile(values, [100*alpha/2, 100*(1-alpha/2)])
        summary[key] = {"mean": float(values.mean()), "ci": (float(lo), float(hi))}
    return summary

def bootstrapping_all_models(
        X,
        y,
        classifiers,
        n_iterations=1000,
        random_state=42,
        ci=0.95) -> dict:
    """Perform bootstrapping on a dataset (val or test) for a list of classifiers.
    Args:
        X (DataFrame): Input features for the dataset to bootstrap
        y (Series): Target variable for the dataset
        classifiers: list of trained models (dicts with key 'classifier')
        n_iterations (int): Number of bootstrap iterations
        random_state (int): Random seed for reproducibility
        ci (float): Confidence interval for the bootstrap estimates
    Returns:
        results (dict): Dictionary of performance metrics summary (mean + CI) for each model
    Note:
    Not used in the current version, as we are using the joint_bootstrap_metrics for relative model comparison,
    but can be used for absolute performance estimation on a single dataset without model comparison.
    """

    results = {}

    for idx, clf in enumerate(classifiers):
        model_name = list(clf["classifier"].named_steps.keys())[-1]
        seed = random_state + idx * 10000  # Different seed for each model

        metric_list = bootstrapping_fast(
            X,
            y,
            clf["classifier"],
            n_iterations=n_iterations,
            random_state=seed,
        )
        results[model_name] = summarize_bootstrap_list(metric_list, ci=ci)
    return results

def bootstrapping_fast(
    X,
    y,
    model,
    n_iterations=1000,
    random_state=42,
):
    """This function performs bootstrapping to estimate the variability of model performance metrics.
    To make it more efficient the inference is ran once on the full dataset and
    then the bootstrap samples are generated using indices, avoiding repeated predictions in each iteration.
    Args:
        X (DataFrame): Input features (options: X_val, X_test)
        y (Series): Target variable (options: y_val, y_test)
        model: Trained classifier for which to perform bootstrapping
        n_iterations (int): Number of bootstrap iterations
        random_state (int): Random seed for reproducibility
    Returns:
        metrics_list (list): List of performance metrics for each bootstrap sample
    Note:
    Not used in the current version, as we are using the joint_bootstrap_metrics for relative model comparison,
    but can be used for absolute performance estimation on a single dataset without model comparison.
    """
    
    # Use random number generator for reproducibility (use a local RNG to avoid global random-state side effects).
    rng = np.random.default_rng(random_state)
    N = len(y)

    # Predict on the full dataset once to avoid repeated predictions in each bootstrap iteration and convert to numpy array if it's a pandas Series or DataFrame
    y_pred_full = model.predict(X)
    if hasattr(y_pred_full, "to_numpy"):
        y_pred_full = y_pred_full.to_numpy()

    metrics_list = []

    # Generate bootstrap samples and compute metrics in a loop, using the precomputed predictions
    for _ in range(n_iterations):
        idx = rng.integers(0, N, size=N)

        y_true_b = y.iloc[idx] if hasattr(y, "iloc") else y[idx]
        y_pred_b = y_pred_full[idx]

        report = compute_metrics(y_true_b, y_pred_b)
        metrics_list.append(extract_global_metrics(report))

    return metrics_list

def bootstrapping_slow(X, y, model, n_iterations=1000, random_state=42):
    """This function performs bootstrapping on the test dataset to estimate the variability of model performance metrics 
    using sklearn's resample function, which is less efficient than the custom implementation in bootstrapping_test_fast
    but more straightforward and compatible with pandas DataFrames and Series.
    Args:
        X_test (DataFrame): Test input features
        y_test (Series): Test target variable
        model: Trained model
        n_iterations (int): Number of bootstrap iterations
        random_state (int): Random seed for reproducibility
    Returns:
        metrics_list (list): List of performance metrics for each bootstrap sample
    Note:
    Not used in the current version, as we are using the joint_bootstrap_metrics for relative model comparison,
    but can be used for absolute performance estimation on a single dataset without model comparison.
    """
    metrics_list = []
    N = len(X)

    for i in range(n_iterations):
        # Create a bootstrap sample
        X_resampled, y_resampled = resample(
            X,
            y,
            replace=True,
            n_samples=N,
            random_state=random_state + i,
        )

        # Make predictions on the bootstrap sample
        y_pred = model.predict(X_resampled)

        # Compute performance metrics
        report = compute_metrics(y_resampled, y_pred)
        metrics_list.append(extract_global_metrics(report))
    return metrics_list