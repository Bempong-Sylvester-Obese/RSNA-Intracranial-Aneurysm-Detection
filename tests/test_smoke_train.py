"""Fast CPU smoke test for training loop (synthetic data)."""

from pathlib import Path

from rsna_aneurysm.config import TrainConfig
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
