import numpy as np # type: ignore
from sklearn.metrics import cohen_kappa_score, accuracy_score # type: ignore


def strict_accuracy(y_true, y_pred) -> float:
    return accuracy_score(y_true, y_pred)


def weighted_kappa(y_true, y_pred, weights: str = "quadratic") -> float:
    """
    Rubric-aligned agreement metric for ordinal labels.
    Quadratic weights penalize larger disagreements more.
    """
    return cohen_kappa_score(y_true, y_pred, weights=weights)


def adjacent_accuracy(y_true, y_pred) -> float:
    """
    Counts prediction as correct if exact OR off by +/- 1.
    Mirrors many rubric-based decision tolerances.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred) <= 1))
