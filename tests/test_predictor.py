"""Predictor returns all label columns for a preprocessed volume."""

from __future__ import annotations

import numpy as np
import torch

from rsna_aneurysm.inference.predictor import PredictionResult, Predictor
from rsna_aneurysm.labels import LABEL_COLUMNS, NUM_LABELS, PRESENCE_COL


def test_prediction_result_from_probs() -> None:
    probs = torch.zeros(NUM_LABELS)
    probs[0] = 0.25
    probs[-1] = 0.75
    r = PredictionResult.from_probs_tensor(probs)
    assert r.presence == r.labels[PRESENCE_COL] == 0.75
    assert len(r.labels) == NUM_LABELS
    for name in LABEL_COLUMNS:
        assert name in r.labels


def test_predictor_random_volume_cpu() -> None:
    device = torch.device("cpu")
    p = Predictor.load(None, device)
    vol = np.random.RandomState(0).rand(16, 128, 128).astype(np.float32)
    r = p.predict_volume(vol)
    assert len(r.labels) == NUM_LABELS
    assert 0.0 <= r.presence <= 1.0
