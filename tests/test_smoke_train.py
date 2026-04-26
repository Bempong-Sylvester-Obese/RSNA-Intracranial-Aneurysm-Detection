"""Fast CPU smoke test for training loop (synthetic data)."""

from pathlib import Path

import pandas as pd
import torch

from rsna_aneurysm.cli import predict
from rsna_aneurysm.config import TrainConfig
from rsna_aneurysm.labels import LABEL_COLUMNS
from rsna_aneurysm.models.aneurysm_net import EfficientAneurysmNet
from rsna_aneurysm.training.trainer import run_training


def test_synthetic_training_smoke(tmp_path: Path) -> None:
    cfg = TrainConfig(
        synthetic=True,
        synthetic_samples=32,
        epochs=1,
        batch_size=4,
        n_folds=2,
        current_fold=0,
        checkpoint_dir=tmp_path / "ckpt",
        runs_dir=tmp_path / "runs",
        early_stopping_patience=99,
        seed=0,
    )
    out = run_training(cfg)
    assert len(out) >= 1
    assert (tmp_path / "ckpt" / "training_summary.json").is_file()


def test_predict_writes_submission_csv(tmp_path: Path) -> None:
    series_dir = tmp_path / "series"
    series_dir.mkdir()

    test_csv = tmp_path / "test.csv"
    pd.DataFrame({"SeriesInstanceUID": ["1.2.3"]}).to_csv(test_csv, index=False)

    ckpt_path = tmp_path / "model.pt"
    model = EfficientAneurysmNet()
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    out_csv = tmp_path / "submission.csv"
    predict(
        checkpoint=ckpt_path,
        test_csv=test_csv,
        series_dir=series_dir,
        out=out_csv,
        batch_size=1,
        verbose=False,
    )

    assert out_csv.is_file()
    written = pd.read_csv(out_csv)
    assert list(written.columns) == ["SeriesInstanceUID", *LABEL_COLUMNS]
