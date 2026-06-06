import numpy as np
from sklearn.metrics import f1_score


def find_best_threshold(
    y_true,
    probs
):
    """
    Finds the optimal classification threshold
    using F1 score maximization.
    """

    thresholds = np.linspace(0.01, 0.99, 1000)

    f1_scores = []

    for t in thresholds:

        preds = (probs > t).astype(int)

        f1_scores.append(
            f1_score(y_true, preds)
        )

    best_threshold = thresholds[np.argmax(f1_scores)]

    return best_threshold