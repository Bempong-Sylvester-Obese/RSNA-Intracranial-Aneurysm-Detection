"""Weighted columnwise macro-averaged AUCROC"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import pandas.api.types
from sklearn.metrics import roc_auc_score


class ParticipantVisibleError(Exception):
    pass


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    class_weights: Optional[List[float]] = None,
) -> float:
    """Compute weighted macro-average multilabel AUCROC."""
    solution = solution.copy()
    submission = submission.copy()
    if row_id_column_name not in solution.columns or row_id_column_name not in submission.columns:
        raise ParticipantVisibleError(
            f"Both solution and submission must include '{row_id_column_name}'."
        )

    del solution[row_id_column_name]
    del submission[row_id_column_name]

    for col in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f"Submission column {col} must be numeric.")

    if len(solution.columns) != len(submission.columns):
        raise ParticipantVisibleError("Submission must have predictions for every class.")

    if list(solution.columns) != list(submission.columns):
        raise ParticipantVisibleError(
            "Submission columns must exactly match solution columns in name and order."
        )

    return float(
        weighted_multilabel_auc(solution.values, submission.values, class_weights)
    )


def weighted_multilabel_auc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    class_weights: Optional[List[float]] = None,
) -> float:
    """Weighted average of per-class ROC AUCs."""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    n_classes = y_true.shape[1]

    try:
        individual_aucs = roc_auc_score(y_true, y_scores, average=None)
    except ValueError:
        raise ParticipantVisibleError(
            "AUC could not be calculated from given predictions."
        ) from None

    if class_weights is None:
        weights_array = np.ones(n_classes)
    else:
        weights_array = np.asarray(class_weights)

    if len(weights_array) != n_classes:
        raise ValueError(
            f"Number of weights ({len(weights_array)}) must match "
            f"number of classes ({n_classes})"
        )

    if np.any(weights_array < 0):
        raise ValueError("All class weights must be non-negative")

    if np.sum(weights_array) == 0:
        raise ValueError("At least one class weight must be positive")

    weights_array = weights_array / np.sum(weights_array)
    return float(np.sum(individual_aucs * weights_array))
