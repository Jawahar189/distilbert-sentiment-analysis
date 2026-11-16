"""Classification metrics: accuracy, precision, recall, and F1."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(eval_pred) -> dict[str, float]:
    """Hugging Face Trainer-compatible metric callback."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, average="binary")),
        "recall": float(recall_score(labels, predictions, average="binary")),
        "f1": float(f1_score(labels, predictions, average="binary")),
    }
