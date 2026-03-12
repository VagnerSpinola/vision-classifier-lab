from __future__ import annotations

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_classification_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
    }