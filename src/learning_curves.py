import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve


def plot_learning_curves(
    best_model,
    X_train,
    y_train,
):
    """TODO: decide if and where to use them"""
    train_sizes, train_scores, val_scores = learning_curve(
        best_model,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
    )
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Training Score")
    plt.plot(train_sizes, np.mean(val_scores, axis=1), label="Validation Score")
    plt.xlabel("Number of Training Examples")
    plt.ylabel("Accuracy")
    plt.title("Learning Curves")
    plt.legend()
    plt.show()
