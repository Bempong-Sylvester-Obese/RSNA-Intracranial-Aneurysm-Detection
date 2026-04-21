"""Tests for competition metric (notebook examples)."""

import numpy as np
import pandas as pd
import pytest

from rsna_aneurysm.metrics import ParticipantVisibleError, score, weighted_multilabel_auc


def test_perfect_score():
    solution = pd.DataFrame({"id": [1, 2, 3], "cat": [1, 0, 1], "dog": [0, 1, 1]})
    submission = pd.DataFrame(
        {"id": [1, 2, 3], "cat": [0.9, 0.2, 0.8], "dog": [0.1, 0.7, 0.6]}
    )
    assert score(solution.copy(), submission.copy(), "id") == pytest.approx(1.0)


def test_weighted_score():
    solution = pd.DataFrame({"id": [1, 2, 3, 4], "A": [0, 0, 1, 1], "B": [0, 0, 1, 1]})
    submission = pd.DataFrame(
        {"id": [1, 2, 3, 4], "A": [0.1, 0.2, 0.8, 0.9], "B": [0.8, 0.2, 0.1, 0.9]}
    )
    assert score(solution.copy(), submission.copy(), "id") == pytest.approx(0.75)
    s31 = score(solution.copy(), submission.copy(), "id", class_weights=[3, 1])
    assert s31 == pytest.approx(0.875)
    s13 = score(solution.copy(), submission.copy(), "id", class_weights=[1, 3])
    assert s13 == pytest.approx(0.625)


def test_weighted_multilabel_auc_uniform():
    y_true = np.array([[1, 0], [0, 1], [1, 1]])
    y_score = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.6]])
    auc = weighted_multilabel_auc(y_true, y_score, None)
    assert 0.0 <= auc <= 1.0


def test_invalid_submission_columns():
    solution = pd.DataFrame({"id": [1], "a": [1]})
    submission = pd.DataFrame({"id": [1], "a": ["x"]})
    with pytest.raises(ParticipantVisibleError):
        score(solution, submission, "id")


def test_mismatched_submission_column_names():
    solution = pd.DataFrame({"id": [1, 2], "a": [1, 0], "b": [0, 1]})
    submission = pd.DataFrame({"id": [1, 2], "a": [0.9, 0.2], "c": [0.1, 0.8]})
    with pytest.raises(ParticipantVisibleError, match="exactly match"):
        score(solution, submission, "id")
